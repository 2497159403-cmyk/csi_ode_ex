"""Frozen endpoint CF sweep. No formal samples are drawn by freeze.

Atomic per-unit commits, cumulative wall-clock budget, and a full pre-test
verification barrier keep fitting and scoring separate. See experiments/README.md for commands.
"""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time
import traceback
import uuid

# Set numerical-library thread ceilings before importing NumPy/SciPy/Torch.
for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '4'

import numpy as np
import scipy
from scipy.integrate import quad
import torch

from .cosine import CosineRankOneField, cosine_features, fit_cosine_rank_one
from .data import EndpointBatch, sample_endpoints
from .io import atomic_json, atomic_torch_save, atomic_csv, derive_seed, hash_json, read_json, sha256_file
from .objectives import LOCATION_A as A, LOCATION_C as C, LOCATION_D as D
from .solvers import euler
from .truth import Truth, schedule

PROJECT = Path(__file__).resolve().parents[1]
CORE_SHA = '2f6c81ff624938999dfab0782582ed91f86609d5e9b0dfc0834d5a52268c1baf'
FORMAL_SEEDS = list(range(2026100301, 2026100311))
STREAMS = ('train_pool', 'validation_pool', 'test', 'permutation', 'initial')
TOLERANCES = {'independent_objective_abs': 2e-10, 'formula_max_abs': 2e-11,
              'decomposition_max_abs': 2e-11, 'ratio_max_abs': 2e-12}


def design(purpose):
    if purpose != 'formal':
        raise ValueError('purpose must be formal')
    return {'version': 1, 'purpose': purpose,
        'namespace': 'cf_endpoint_sweep_v1',
        'M_values': [500, 1000, 2000, 4000, 8000],
        'seeds': FORMAL_SEEDS,
        'n_test': 2048, 'N': 8, 'd': 8, 'val_fraction': .1,
        'n_features': 13, 'rank': 1, 'lambda': 0, 't_fixed': .5,
        'K_values': [10, 20, 40, 80, 160], 'euler_M': 4000,
        'runtime': {'device': 'cpu', 'num_threads': 4, 'fit_dtype': 'float64',
                    'parameter_dtype': 'float32', 'training_data_dtype': 'float32', 'test_data_dtype': 'float64',
                    'exact_protocol': 'float32 parameters promoted to double; float32-rounded x/z0'},
        'truth': {'u': [1, 0, 0, 0, 0, 0, 0, 0],
                  'bump': {'J': 3, 'beta': 2, 'delta': 1, 'omega': [1, 0, 1]},
                  'factor_budget': 2, 'activation_bound': 2, 'slope_bound': 32, 'index_scale': 1},
        'budget': {'active_seconds': 1800, 'fit_seconds': 60},
        'tolerances': TOLERANCES, 'statistics_unit': 'base_seed',
        'validation_role': 'report noisy loss only; never select',
        'test_barrier': 'all committed solutions and pools verified before test access'}


def validate_config(cfg):
    if hash_json(cfg) != hash_json(design(cfg.get('purpose'))):
        raise ValueError('Configuration differs from the frozen design; no overrides or seed selection')
    return cfg


def truth_for(cfg, dtype='float64'):
    return Truth({'protocol': 'location', 'data': {'N': cfg['N'], 'd': cfg['d']},
                  'truth': cfg['truth'], 'model': {'rank': 1}, 'runtime': {'dtype': dtype}})


def safe_root(root):
    root=Path(root).absolute()
    allowed=(PROJECT/'outputs').resolve()
    resolved=root.resolve()
    if resolved==allowed or not resolved.is_relative_to(allowed):
        raise ValueError('Run must be a new child directory inside this project outputs; external paths forbidden')
    for parent in (root,*root.parents):
        if parent==PROJECT:break
        if parent.is_symlink():raise ValueError('Symlink run paths are forbidden')
    return resolved


def source_files():
    paths = [p for folder in ('endpoint_ode', 'configs')
             for p in (PROJECT / folder).glob('*') if p.suffix in ('.py', '.json')]
    paths += [PROJECT / 'requirements.txt']
    return {str(p.relative_to(PROJECT)): sha256_file(p) for p in sorted(paths)}


def runtime_identity():
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform(),
            'device': 'cpu', 'threads': 4, 'deterministic': True}


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def freeze(config, root):
    configure()
    cfg = validate_config(read_json(config))
    root = safe_root(root)
    if root.exists():
        raise FileExistsError('Run directory already exists; immutable freeze')
    files = source_files()
    if files['endpoint_ode/cosine.py'] != CORE_SHA:
        raise ValueError('Protected cosine core changed')
    seeds = {str(s): {p: derive_seed(s, cfg['namespace'], p) for p in STREAMS} for s in cfg['seeds']}
    flat = [v for row in seeds.values() for v in row.values()]
    if len(flat) != len(set(flat)):
        raise ValueError('Derived random stream collision')
    plan = {'config': cfg, 'source': files, 'environment': runtime_identity(), 'seeds': seeds,
            'namespace_check': 'Source baseline scanned before implementation; namespace absent. New run seeds fixed.',
            'seed_rule': 'derive_seed(base_seed, namespace, stream)',
            'expected_fits': len(cfg['seeds']) * len(cfg['M_values'])}
    root.mkdir(parents=True)
    for rel in files:
        dst = root / 'source_snapshot' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / rel, dst)
    atomic_json(root / 'plan.json', plan)
    atomic_json(root / 'identity.json', {'plan_sha256': sha256_file(root / 'plan.json'), 'identity': hash_json(plan)})
    atomic_json(root / 'progress.json', {'active_seconds': 0., 'events': [], 'status': 'frozen'})
    return plan


def open_run(root):
    configure()
    root = safe_root(root)
    plan = read_json(root / 'plan.json')
    ident = read_json(root / 'identity.json')
    if ident != {'plan_sha256': sha256_file(root / 'plan.json'), 'identity': hash_json(plan)}:
        raise ValueError('Plan identity changed')
    validate_config(plan['config'])
    if plan['source'] != source_files() or plan['environment'] != runtime_identity():
        raise ValueError('Source/config/dependency identity changed; use a new run')
    for rel, digest in plan['source'].items():
        if sha256_file(root / 'source_snapshot' / rel) != digest:
            raise ValueError('Frozen source snapshot changed')
    return root, plan, ident['identity']


def batch_dict(b):
    return {k: getattr(b, k) for k in ('x', 'y0', 'y1', 'eta', 'ids')}


def tensor_digest(obj):
    h = hashlib.sha256()
    for k in ('x', 'y0', 'y1', 'eta'):
        t = obj[k].contiguous().cpu()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    h.update(json.dumps(obj['ids']).encode())
    return h.hexdigest()


def unit_load(root, rel, identity):
    folder = root / rel
    if not (folder / 'commit.json').exists():
        return None
    commit = read_json(folder / 'commit.json')
    if commit['identity'] != identity:
        raise ValueError(f'Unit identity changed: {rel}')
    for name, sha in commit['files'].items():
        if sha256_file(folder / name) != sha:
            raise ValueError(f'Committed artifact changed: {rel}/{name}')
    return commit


def prepare_unit(root, rel, identity):
    folder = root / rel
    if unit_load(root, rel, identity):
        raise FileExistsError(f'Committed unit is immutable: {rel}')
    if folder.exists():
        # Preserve interrupted, uncommitted work; never overwrite it silently.
        dst = root / 'interrupted' / (rel.replace('/', '_') + '_' + uuid.uuid4().hex)
        dst.parent.mkdir(exist_ok=True)
        folder.rename(dst)
    folder.mkdir(parents=True)
    return folder


def commit_unit(folder, identity, metadata=None):
    files = {p.name: sha256_file(p) for p in sorted(folder.iterdir()) if p.is_file()}
    atomic_json(folder / 'commit.json', {'identity': identity, 'files': files, 'metadata': metadata or {}})


def payload(root, rel, identity):
    if unit_load(root, rel, identity) is None:
        raise ValueError(f'Missing committed unit: {rel}')
    return torch.load(root / rel / 'data.pt', map_location='cpu', weights_only=False)


def pools(root, plan, identity, seed, create=False):
    rel = f'pools/{seed}'
    if unit_load(root, rel, identity):
        return payload(root, rel, identity)
    if not create:
        raise ValueError(f'Missing training pools: {seed}')
    cfg = plan['config']; truth = truth_for(cfg,'float32'); M = max(cfg['M_values'])
    data = {}
    for kind, count in (('train_pool', 9*M//10), ('validation_pool', M//10)):
        data[kind] = batch_dict(sample_endpoints(truth, count, plan['seeds'][str(seed)][kind],
                                                f"{cfg['namespace']}:{seed}:{kind}"))
    folder = prepare_unit(root, rel, identity)
    atomic_torch_save(folder / 'data.pt', data)
    atomic_json(folder / 'data_identity.json', {k: tensor_digest(v) for k, v in data.items()})
    commit_unit(folder, identity)
    return data


def subset(data, count):
    return EndpointBatch(**data).subset(slice(0, count))


def complete_loss(field, batch):
    batch=batch.to(dtype=torch.float64)
    r = field.amplitude(batch.x, batch.y0)
    return float((A*(r-batch.y1-C/A*batch.y0).square()+(D-C*C/A)*batch.y0.square()).mean())


def load_field(saved, dtype=torch.float64):
    f = CosineRankOneField(8).to(dtype=dtype)
    f.load_state_dict({k: v.to(dtype=dtype) for k, v in saved.items()})
    return f


def fit(root, resume=False):
    root, plan, identity = open_run(root); cfg = plan['config']
    existing = any((root / 'fits').glob('*/*/commit.json')) if (root / 'fits').exists() else False
    if existing and not resume:
        raise FileExistsError('Fits already exist; use resume to verify and skip completed fits')
    for seed in cfg['seeds']:
        data = pools(root, plan, identity, seed, create=True)
        for M in cfg['M_values']:
            rel = f'fits/{seed}/{M}'
            if unit_load(root, rel, identity):
                payload(root, rel, identity)
                continue
            folder = prepare_unit(root, rel, identity)
            train = subset(data['train_pool'], 9*M//10); val = subset(data['validation_pool'], M//10)
            start = time.monotonic()
            old_timer = signal.getitimer(signal.ITIMER_REAL)
            signal.setitimer(signal.ITIMER_REAL, min(cfg['budget']['fit_seconds'], old_timer[0] or math.inf))
            try:
                field, report, evidence = fit_cosine_rank_one(train.x, train.y0, train.y1)
            finally:
                signal.setitimer(signal.ITIMER_REAL, max(.001, old_timer[0]-(time.monotonic()-start)) if old_timer[0] else 0)
            seconds = time.monotonic()-start
            runtime = copy.deepcopy(field).float()
            mean64 = train.y1.double().mean((0, 1))
            result = {'parameters64': field.state_dict(), 'parameters32': runtime.state_dict(),
                      'mean64': mean64, 'mean32': mean64.float(), 'solver': report, 'evidence': evidence,
                      'M': M, 'seed': seed, 'train_ids': train.ids, 'validation_ids': val.ids,
                      'train_digest': tensor_digest(batch_dict(train)), 'validation_digest': tensor_digest(batch_dict(val)),
                      'train_loss_per_coordinate': complete_loss(field, train),
                      'validation_loss_per_coordinate': complete_loss(field, val), 'fit_seconds': seconds,
                      'coefficient_bounds': {'amplitude_norm': float((field.beta[0].abs()+math.sqrt(2)*field.beta[1:].abs().sum())*field.direction.norm()),
                          'x_derivative_norm': float(math.sqrt(2)*math.pi*(torch.arange(1,13)*field.beta[1:].abs()).sum()*field.direction.norm()),
                          'state_Lipschitz': 0., 'theory_budgets_enforced': False}}
            atomic_torch_save(folder / 'data.pt', result)
            atomic_json(folder / 'report.json', {k: v for k, v in result.items() if k not in ('parameters64','parameters32','mean64','mean32','evidence')})
            commit_unit(folder, identity, {'seed': seed, 'M': M, 'seconds': seconds})
    barrier(root, plan, identity)
    return {'status': 'all_fits_committed', 'count': plan['expected_fits']}


def barrier(root, plan, identity):
    solutions = {}
    for seed in plan['config']['seeds']:
        data = pools(root, plan, identity, seed)
        for M in plan['config']['M_values']:
            rel = f'fits/{seed}/{M}'
            item = payload(root, rel, identity)
            for name, kind, n in (('train', 'train_pool', 9*M//10), ('validation', 'validation_pool', M//10)):
                b = subset(data[kind], n)
                if item[name+'_digest'] != tensor_digest(batch_dict(b)) or item[name+'_ids'] != b.ids:
                    raise ValueError('Solution/data identity mismatch')
            solutions[f'{seed}/{M}'] = sha256_file(root / rel / 'data.pt')
    return solutions


def evaluation_data(root, plan, identity, seed):
    # Public helper independently enforces the barrier even outside evaluate().
    committed = barrier(root, plan, identity)
    rel = f'test/{seed}'
    if unit_load(root, rel, identity):
        return payload(root, rel, identity)
    cfg = plan['config']; stream = plan['seeds'][str(seed)]
    b = sample_endpoints(truth_for(cfg), cfg['n_test'], stream['test'], f"{cfg['namespace']}:{seed}:test")
    g = torch.Generator().manual_seed(stream['initial'])
    data = {'batch': batch_dict(b), 'z0': torch.randn(b.y0.shape, generator=g, dtype=torch.float64).float(),
            'all_solutions_before_test': committed,
            'permutation': torch.randperm(cfg['n_test'], generator=torch.Generator().manual_seed(stream['permutation']))}
    folder = prepare_unit(root, rel, identity)
    atomic_torch_save(folder / 'data.pt', data)
    atomic_json(folder / 'data_identity.json', {'test_digest': tensor_digest(data['batch']), 'solutions': committed})
    commit_unit(folder, identity)
    return data


class ConstantField:
    def __init__(self, mean): self.mean = mean
    def amplitude(self, x, z): return self.mean.to(z)[None,None,:].expand_as(z)
    def __call__(self, t, x, z):
        return schedule('location', torch.as_tensor(t, dtype=z.dtype))['db']*self.amplitude(x,z)


def units(v):
    raw = v.double().square().sum((1,2))
    return {'raw': raw, 'per_token': raw/v.shape[1], 'per_coordinate': raw/(v.shape[1]*v.shape[2])}


def per_token_inner(a, b): return 2*(a*b).sum((1,2))/a.shape[1]


def numerical(field64, field32, x64, z64, target, K):
    amplitude = field64.amplitude(x64, z64)
    exact = z64+amplitude
    double_euler = euler(field64, x64, z64, K)
    runtime = euler(field32, x64.float(), z64.float(), K).double()
    q = sum(math.cos(math.pi*(k/K)*(k/K-.5))*math.pi*(2*k/K-.5) for k in range(K))/K
    formula = z64+q*amplitude
    discretization = double_euler-exact; rounding = runtime-double_euler; total = runtime-exact
    learned_error = exact-target
    out = {'q': q, 'formula_max_abs': float((formula-double_euler).abs().max()),
           'discretization': units(discretization), 'rounding': units(rounding), 'total': units(total),
           'rounding_cross': per_token_inner(discretization,rounding),
           'risk_cross_double': per_token_inner(learned_error,discretization),
           'risk_cross_runtime': per_token_inner(learned_error,total),
           'exact_risk': units(learned_error), 'euler_risk_double': units(double_euler-target),
           'euler_risk_runtime': units(runtime-target)}
    if out['formula_max_abs'] > TOLERANCES['formula_max_abs']:
        raise AssertionError('Euler implementation does not match independent qK formula')
    for lhs, rhs in ((out['total']['per_token'], out['discretization']['per_token']+out['rounding']['per_token']+out['rounding_cross']),
                     (out['euler_risk_double']['per_token'], out['exact_risk']['per_token']+out['discretization']['per_token']+out['risk_cross_double']),
                     (out['euler_risk_runtime']['per_token'], out['exact_risk']['per_token']+out['total']['per_token']+out['risk_cross_runtime'])):
        if float((lhs-rhs).abs().max()) > TOLERANCES['decomposition_max_abs']:
            raise AssertionError('Numerical error decomposition failed')
    return out


class OracleField:
    def __init__(self, truth): self.truth = truth
    def amplitude(self, x, z): return (self.truth.f(x)[:,:,None]*self.truth.u.to(z)).expand_as(z)
    def __call__(self, t, x, z): return self.truth.drift(t,x,z)


def evaluate(root, resume=False):
    root, plan, identity = open_run(root); cfg = plan['config']
    barrier(root, plan, identity)  # verify every fit before creating or reading ANY test
    if unit_load(root, 'evaluation', identity):
        if resume:
            verify_evaluation(root,plan,identity)
            return read_json(root/'evaluation/summary.json')
        raise FileExistsError('Evaluation is immutable; use resume')
    rows = []; numerical_rows = []
    for seed in cfg['seeds']:
        test = evaluation_data(root, plan, identity, seed)
        b = EndpointBatch(**test['batch']); x = b.x.float().double(); z = test['z0'].double()
        truth = truth_for(cfg); target = truth.flow(1., x, z)
        for M in cfg['M_values']:
            rel = f'scores/{seed}/{M}'
            if unit_load(root, rel, identity):
                out = payload(root, rel, identity)
                rows.extend(out['rows']); numerical_rows.extend(out['numerical_rows']); continue
            solved = payload(root, f'fits/{seed}/{M}', identity)
            cf32 = load_field(solved['parameters32'],torch.float32); cf64 = load_field(solved['parameters32'])
            pairs = {'CF': (cf64,cf32), 'train_mean': (ConstantField(solved['mean32'].double()),ConstantField(solved['mean32'])),
                     'zero': (ConstantField(torch.zeros(8,dtype=torch.float64)),ConstantField(torch.zeros(8)))}
            vectors = {}; local_rows = []; local_num = []
            for name, (f64,f32) in pairs.items():
                exact = z+f64.amplitude(x,z)
                terminal = units(exact-target)
                truefield = truth.drift(.5,x,z)
                field = units(f64(.5,x,z)-truefield)
                runtimefield = units(f32(.5,x.float(),z.float()).double()-truefield)
                residual = field['per_token']-(math.pi**2/4)*terminal['per_token']
                if float(residual.abs().max()) > TOLERANCES['ratio_max_abs']:
                    raise AssertionError('Field/terminal same-precision identity failed')
                vectors[name] = {'terminal':terminal,'field':field,'runtime_field':runtimefield,
                    'ratio_residual':residual,'runtime_ratio_residual':runtimefield['per_token']-(math.pi**2/4)*terminal['per_token']}
                row = {'seed':seed,'M':M,'n_fit':9*M//10,'n_validation':M//10,'method':name}
                for metric in ('terminal','field','runtime_field'):
                    for unit,v in vectors[name][metric].items(): row[f'{metric}_{unit}'] = float(v.mean())
                row.update(ratio_max_abs=float(residual.abs().max()),runtime_ratio_max_abs=float(vectors[name]['runtime_ratio_residual'].abs().max()))
                local_rows.append(row)
            if M == cfg['euler_M']:
                pairs['oracle_numerical_reference'] = (OracleField(truth),OracleField(truth))
                vectors['numerical'] = {}
                for name,(f64,f32) in pairs.items():
                    vectors['numerical'][name] = {}
                    for K in cfg['K_values']:
                        num = numerical(f64,f32,x,z,target,K); vectors['numerical'][name][K] = num
                        row = {'seed':seed,'M':M,'method':name,'K':K,'q':num['q'],'formula_max_abs':num['formula_max_abs']}
                        for key in ('discretization','rounding','total','exact_risk','euler_risk_double','euler_risk_runtime'):
                            row[key] = float(num[key]['per_token'].mean())
                        for key in ('rounding_cross','risk_cross_double','risk_cross_runtime'): row[key] = float(num[key].mean())
                        local_num.append(row)
            out = {'test_ids':b.ids,'vectors':vectors,'rows':local_rows,'numerical_rows':local_num}
            folder = prepare_unit(root,rel,identity); atomic_torch_save(folder/'data.pt',out); commit_unit(folder,identity)
            rows.extend(local_rows); numerical_rows.extend(local_num)
    folder = prepare_unit(root,'evaluation',identity)
    atomic_csv(folder/'per_seed.csv',rows); atomic_csv(folder/'euler.csv',numerical_rows)
    summary = summarize(rows,cfg)
    atomic_json(folder/'summary.json',summary); atomic_csv(folder/'summary.csv',summary['rows'])
    commit_unit(folder,identity)
    return summary


def summarize(rows,cfg):
    out = []
    for M in cfg['M_values']:
        mean = np.array([next(r['terminal_per_token'] for r in rows if r['M']==M and r['seed']==s and r['method']=='train_mean') for s in cfg['seeds']])
        cf = np.array([next(r['terminal_per_token'] for r in rows if r['M']==M and r['seed']==s and r['method']=='CF') for s in cfg['seeds']])
        for method in ('CF','train_mean','zero'):
            chosen = [next(r for r in rows if r['M']==M and r['seed']==s and r['method']==method) for s in cfg['seeds']]
            risk = np.array([r['terminal_per_token'] for r in chosen]); field = np.array([r['field_per_token'] for r in chosen])
            out.append({'M':M,'method':method,'n_seeds':len(risk),'terminal_mean':float(risk.mean()),'terminal_sample_SD':float(risk.std(ddof=1)),
                'field_mean':float(field.mean()),'field_sample_SD':float(field.std(ddof=1)),
                'wins_train_mean':int((risk<mean).sum()),'paired_difference_mean':float((risk-mean).mean()),
                'paired_difference_sample_SD':float((risk-mean).std(ddof=1)),
                'relative_improvement_ratio_of_means':float(1-risk.mean()/mean.mean()),
                'ratio_residual_max':max(r['ratio_max_abs'] for r in chosen)})
    ratios = [{'M':M,'seed':s,'CF_over_train_mean':next(r['terminal_per_token'] for r in rows if r['M']==M and r['seed']==s and r['method']=='CF')/next(r['terminal_per_token'] for r in rows if r['M']==M and r['seed']==s and r['method']=='train_mean')} for M in cfg['M_values'] for s in cfg['seeds']]
    return {'purpose':cfg['purpose'],'rows':out,'per_seed_ratios':ratios,
            'interpretation':'Finite-test coupling MSE; seed SD includes training and test randomness.'}


def independent_fit(batch):
    x=batch.x.double().numpy(); y0=batch.y0.double().numpy(); y1=batch.y1.double().numpy()
    phi=np.cos(math.pi*x*np.arange(13)); phi[:,1:]*=math.sqrt(2)
    effective=y1+(C/A)*y0
    q,r=np.linalg.qr(phi,mode='reduced')
    u,s,vh=np.linalg.svd(q.T@effective.mean(1),full_matrices=False)
    W=np.linalg.solve(r,np.outer(u[:,0]*s[0],vh[0]))
    loss=np.mean(A*((phi@W)[:,None,:]-effective)**2+(D-C*C/A)*y0**2)
    return phi,W,float(loss),s


def verify_evaluation(root,plan,identity):
    committed=barrier(root,plan,identity)
    for seed in plan['config']['seeds']:
        test=payload(root,f'test/{seed}',identity)
        if test['all_solutions_before_test']!=committed:
            raise ValueError('Test/solution commitment mismatch')
        for M in plan['config']['M_values']:
            score=payload(root,f'scores/{seed}/{M}',identity)
            if score['test_ids']!=test['batch']['ids']:
                raise ValueError('Score/test identity mismatch')
    if not unit_load(root,'evaluation',identity):raise ValueError('Missing evaluation commit')


def check_csv(path,rows):
    with open(path,newline='') as stream:stored=list(csv.DictReader(stream))
    if len(stored)!=len(rows):raise AssertionError('CSV row count mismatch')
    for a,b in zip(stored,rows):
        if set(a)!=set(b):raise AssertionError('CSV columns mismatch')
        for key,val in b.items():
            if isinstance(val,(int,float)):
                if not math.isclose(float(a[key]),val,rel_tol=2e-12,abs_tol=2e-12):raise AssertionError(f'CSV numeric mismatch {key}')
            elif a[key]!=str(val):raise AssertionError('CSV label mismatch')


def verify_score_rows(score):
    for row in score['rows']:
        v=score['vectors'][row['method']]
        for metric in ('terminal','field','runtime_field'):
            raw=v[metric]['raw']
            for unit,factor in (('raw',1),('per_token',8),('per_coordinate',64)):
                if not torch.allclose(v[metric][unit],raw/factor,atol=2e-12,rtol=2e-12):raise AssertionError('Metric units mismatch')
                if abs(row[f'{metric}_{unit}']-float(v[metric][unit].mean()))>2e-12:raise AssertionError('Metric row mean mismatch')
        for col,key in (('ratio_max_abs','ratio_residual'),('runtime_ratio_max_abs','runtime_ratio_residual')):
            if abs(row[col]-float(v[key].abs().max()))>2e-12:raise AssertionError('Ratio residual row mismatch')
    for row in score['numerical_rows']:
        v=score['vectors']['numerical'][row['method']][row['K']]
        for key in ('discretization','rounding','total','exact_risk','euler_risk_double','euler_risk_runtime'):
            for unit,factor in (('raw',1),('per_token',8),('per_coordinate',64)):
                if not torch.allclose(v[key][unit],v[key]['raw']/factor,atol=2e-12,rtol=2e-12):raise AssertionError('Numerical units mismatch')
            if abs(row[key]-float(v[key]['per_token'].mean()))>2e-12:raise AssertionError('Numerical row mean mismatch')
        for key in ('rounding_cross','risk_cross_double','risk_cross_runtime'):
            if abs(row[key]-float(v[key].mean()))>2e-12:raise AssertionError('Cross term row mismatch')
        for lhs,rhs in ((v['total']['per_token'],v['discretization']['per_token']+v['rounding']['per_token']+v['rounding_cross']),
                        (v['euler_risk_double']['per_token'],v['exact_risk']['per_token']+v['discretization']['per_token']+v['risk_cross_double']),
                        (v['euler_risk_runtime']['per_token'],v['exact_risk']['per_token']+v['total']['per_token']+v['risk_cross_runtime'])):
            if not torch.allclose(lhs,rhs,atol=2e-11,rtol=2e-11):raise AssertionError('Saved decomposition mismatch')


def audit(root, resume=False):
    root,plan,identity=open_run(root); cfg=plan['config']; barrier(root,plan,identity)
    if not unit_load(root,'evaluation',identity): raise ValueError('Evaluation must be committed before audit')
    if unit_load(root,'audit',identity):
        if resume:
            verify_evaluation(root,plan,identity)
            report=read_json(root/'audit/report.json')
            if report['artifacts_before_after']!=artifact_inventory(root):raise ValueError('Audited artifacts changed')
            return report
        raise FileExistsError('Audit is immutable')
    before = artifact_inventory(root)
    db=lambda t:math.cos(math.pi*t*(t-.5))*math.pi*(2*t-.5)
    da=lambda t:-math.sin(math.pi*t*(t-.5))*math.pi*(2*t-.5)
    integrals=[quad(fn,0,1,epsabs=1e-13,epsrel=1e-13)[0] for fn in (lambda t:db(t)**2,lambda t:da(t)*db(t),lambda t:da(t)**2)]
    if max(abs(a-b) for a,b in zip(integrals,[A,C,D]))>2e-12:raise AssertionError('Independent schedule integral mismatch')
    verify_evaluation(root,plan,identity)
    checks=[]; all_rows=[]; all_numerical_rows=[]
    for seed in cfg['seeds']:
        pool=pools(root,plan,identity,seed); test=evaluation_data(root,plan,identity,seed)
        xb=test['batch']['x'].float().double().numpy(); z=test['z0'].double().numpy()
        phi_test=np.cos(math.pi*xb*np.arange(13));phi_test[:,1:]*=math.sqrt(2)
        centers=(np.arange(3)+.5)/3; s=(xb[...,None]-centers)*3; inside=np.abs(s)<1/3
        bump=np.where(inside,np.exp(1-1/np.where(inside,1-9*s*s,1)),0)
        fx=(bump*np.array([1,0,1])).sum(-1)/9
        true_amp=fx*np.array([1,0,0,0,0,0,0,0])[None,:]
        for M in cfg['M_values']:
            fitdata=payload(root,f'fits/{seed}/{M}',identity); b=subset(pool['train_pool'],9*M//10)
            phi,W,loss,singular=independent_fit(b)
            Wstored=fitdata['evidence']['W'].numpy()
            effective=b.y1.double().numpy()+C/A*b.y0.double().numpy()
            stored_loss=float(np.mean(A*((phi@Wstored)[:,None,:]-effective)**2+(D-C*C/A)*b.y0.double().numpy()**2))
            if abs(stored_loss-loss)>TOLERANCES['independent_objective_abs'] or np.linalg.matrix_rank(Wstored,tol=1e-10)>1:
                raise AssertionError('Independent QR/SVD optimality failure')
            saved=fitdata['parameters32']; beta=saved['beta'].double().numpy(); direction=saved['direction'].double().numpy()
            amplitudes={'CF':(phi_test@beta)[:,None]*direction,'train_mean':np.broadcast_to(fitdata['mean32'].double().numpy(),true_amp.shape),'zero':np.zeros_like(true_amp)}
            score=payload(root,f'scores/{seed}/{M}',identity);verify_score_rows(score)
            all_rows.extend(score['rows']);all_numerical_rows.extend(score['numerical_rows'])
            for method,amp in amplitudes.items():
                err=np.sum((amp-true_amp)**2,axis=1)
                savederr=score['vectors'][method]['terminal']['per_token'].numpy()
                if not np.allclose(err,savederr,atol=2e-12,rtol=2e-11):raise AssertionError('Independent endpoint metric failure')
                if M==cfg['euler_M']:
                    for K in cfg['K_values']:
                        q=sum(db(k/K) for k in range(K))/K
                        predicted=(q-1)**2*np.sum(amp**2,axis=1)
                        actual=score['vectors']['numerical'][method][K]['discretization']['per_token'].numpy()
                        if not np.allclose(predicted,actual,atol=2e-12,rtol=2e-10):raise AssertionError('Independent Euler all-K audit failed')
            if M==cfg['euler_M']:
                for K in cfg['K_values']:
                    q=sum(db(k/K) for k in range(K))/K
                    actual=score['vectors']['numerical']['oracle_numerical_reference'][K]['discretization']['per_token'].numpy()
                    if not np.allclose((q-1)**2*np.sum(true_amp**2,axis=1),actual,atol=2e-12,rtol=2e-10):raise AssertionError('Oracle Euler audit failed')
            checks.append({'seed':seed,'M':M,'objective_abs_error':abs(loss-stored_loss),'rank':int(np.linalg.matrix_rank(Wstored,tol=1e-10)),
                           'top_singular_gap':float(singular[0]-singular[1]),'comparison':'objective and rank; parameter equality not required for ties'})
    if summarize(all_rows,cfg)!=read_json(root/'evaluation/summary.json'):raise AssertionError('Aggregate table mismatch')
    check_csv(root/'evaluation/per_seed.csv',all_rows)
    check_csv(root/'evaluation/euler.csv',all_numerical_rows)
    check_csv(root/'evaluation/summary.csv',summarize(all_rows,cfg)['rows'])
    if before!=artifact_inventory(root):raise AssertionError('Audit mutated artifacts')
    report={'status':'passed','independent_solver':'NumPy reduced QR + SVD, no training solver call',
            'integrals_ACD':integrals,'checks':checks,'all_K':cfg['K_values'],'artifacts_before_after':before,
            'tolerances':TOLERANCES,'purpose':cfg['purpose']}
    folder=prepare_unit(root,'audit',identity);atomic_json(folder/'report.json',report);commit_unit(folder,identity)
    return report


def artifact_inventory(root):
    return {str(p.relative_to(root)):sha256_file(p) for base in ('pools','fits','test','scores','evaluation')
            for p in sorted((root/base).rglob('*')) if p.is_file()}


def execute(command, root, config=None, resume=False):
    root=safe_root(root); start=time.monotonic(); previous=None
    if command=='freeze':return freeze(config,root)
    _,plan,_=open_run(root)
    lock=root/'command.lock'
    fd=os.open(lock,os.O_CREAT|os.O_RDWR,0o600)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd);raise RuntimeError('Run locked by another live command')
    # The kernel releases flock on process death. Read progress only AFTER the
    # lock: two sequential processes can never overwrite a stale budget view.
    progress=read_json(root/'progress.json')
    orphan=progress.pop('inflight',None)
    if orphan:
        orphan.update(status='interrupted',seconds=max(0.,time.time()-orphan['started_unix']),
                      error='Owner process died; elapsed wall time conservatively charged')
        progress['active_seconds']=min(plan['config']['budget']['active_seconds'],progress['active_seconds']+orphan['seconds'])
        progress['events'].append(orphan)
        atomic_json(root/'events'/f'{time.time_ns()}_interrupted.json',orphan)
        atomic_json(root/'progress.json',progress)
    os.ftruncate(fd,0);os.write(fd,str(os.getpid()).encode())
    remaining=plan['config']['budget']['active_seconds']-progress['active_seconds']
    def timeout(signum,frame):raise TimeoutError('Frozen activity/fit wall-clock budget exhausted')
    previous=signal.signal(signal.SIGALRM,timeout)
    event={'command':command,'resume':resume,'started_unix':time.time(),'status':'running'}
    # Reserve start immediately: a killed process is conservatively charged on recovery.
    progress['inflight']=event;atomic_json(root/'progress.json',progress)
    try:
        if remaining<=0:raise TimeoutError('Cumulative activity budget exhausted')
        signal.setitimer(signal.ITIMER_REAL,remaining)
        result={'fit':fit,'resume':lambda r,resume=False:fit(r,True),'evaluate':evaluate,'audit':audit}[command](root,resume=resume)
        event['status']='completed';progress['status']=command+'_completed';return result
    except BaseException as exc:
        event.update(status='failed',error=repr(exc),traceback=traceback.format_exc());progress['status']='failed'
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        event['seconds']=time.monotonic()-start
        event['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        progress['active_seconds']+=event['seconds'];progress['events'].append(event);progress.pop('inflight',None)
        atomic_json(root/'progress.json',progress)
        atomic_json(root/'events'/f"{time.time_ns()}_{command}.json",event)
        fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','fit','resume','evaluate','audit'])
    p.add_argument('--run',required=True);p.add_argument('--config');p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    try:
        result=execute(args.command,args.run,args.config,args.resume)
        print(json.dumps({'command':args.command,'status':'ok','result':result},ensure_ascii=False))
    except Exception as exc:
        # Rejections before the verified execution wrapper (including identity
        # mismatches) are also retained, without altering committed artifacts.
        try:root=safe_root(args.run)
        except ValueError:root=None
        if root is not None and root.is_dir():
            atomic_json(root/'rejections'/f'{time.time_ns()}.json',
                        {'command':args.command,'error':repr(exc),'time_unix':time.time()})
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
