"""Public cosine rank-one location learner; global fit uses observed training only."""
from __future__ import annotations
import math
import torch
from torch import nn
from .objectives import LOCATION_A,LOCATION_C,LOCATION_D

DEGREE=12


def cosine_features(x):
    if x.ndim!=2 or x.shape[1]!=1 or not torch.isfinite(x).all():
        raise ValueError('Expected finite observed x:[B,1]')
    frequencies=torch.arange(1,DEGREE+1,dtype=x.dtype,device=x.device)
    return torch.cat((torch.ones_like(x),math.sqrt(2)*torch.cos(math.pi*x*frequencies[None,:])),1)


class CosineRankOneField(nn.Module):
    """R(x)=<psi(x),beta>u, equivalent to fixed-index pairwise aggregation."""
    variant='location_cosine_rank1'
    time_dependent_factors=False

    def __init__(self,d):
        super().__init__()
        self.d=d
        self.register_buffer('beta',torch.zeros(DEGREE+1,dtype=torch.float64))
        self.register_buffer('direction',torch.zeros(d,dtype=torch.float64))

    def amplitude(self,x,z):
        if z.ndim!=3 or z.shape[-1]!=self.d or x.shape!=(len(z),1) or z.shape[1]<2:
            raise ValueError('Expected x:[B,1], z:[B,N,d], N>=2')
        scalar=cosine_features(x)@self.beta
        return (scalar[:,None,None]*self.direction[None,None,:]).expand_as(z)

    def forward(self,t,x,z):
        t=torch.as_tensor(t,dtype=z.dtype,device=z.device)
        if t.ndim==0:t=t.expand(len(z))
        if t.shape!=(len(z),):raise ValueError('Expected scalar or batched time')
        db=torch.cos(math.pi*t*(t-.5))*(math.pi*(2*t-.5))
        return db[:,None,None]*self.amplitude(x,z)


@torch.no_grad()
def fit_cosine_rank_one(x,y0,y1):
    """Only observable training arrays; no config, validation, Truth or test input.

    The positive common A/d objective scale does not change the lambda-zero
    minimizer. Logs below recompute the full endpoint/token loss with that scale.
    """
    x=x.detach().cpu().double(); y0=y0.detach().cpu().double(); y1=y1.detach().cpu().double()
    if y0.ndim!=3 or y0.shape!=y1.shape or x.shape!=(len(y0),1) or y0.shape[1]<2 or y0.shape[2]<1 or len(y0)<DEGREE+1:
        raise ValueError('Invalid training shapes or insufficient endpoint count')
    if not torch.isfinite(y0).all() or not torch.isfinite(y1).all():raise ValueError('Nonfinite training response')
    phi=cosine_features(x)
    effective=y1+(LOCATION_C/LOCATION_A)*y0
    mean_effective=effective.mean(1)
    G=phi.T@phi/len(phi)
    G=(G+G.T)/2
    eig=torch.linalg.eigvalsh(G)
    if not torch.isfinite(eig).all() or eig[0]<=0:raise ValueError('Cosine Gram is not finite positive definite')
    condition=(eig[-1]/eig[0]).item()
    if condition>1e8:raise ValueError('Cosine Gram condition exceeds frozen 1e8 threshold; no ridge fallback')
    H=phi.T@mean_effective/len(phi)
    L=torch.linalg.cholesky(G)
    T=torch.linalg.solve_triangular(L,H,upper=False)
    left,singular,right=torch.linalg.svd(T,full_matrices=False)
    direction=right[0].clone()
    beta=torch.linalg.solve_triangular(L.T,(left[:,0]*singular[0])[:,None],upper=True).squeeze(1)
    pivot=int(direction.abs().argmax())
    if direction[pivot]<0:direction=-direction;beta=-beta
    field=CosineRankOneField(y0.shape[2])
    field.beta.copy_(beta);field.direction.copy_(direction)
    W=beta[:,None]*direction[None,:]
    prediction=(phi@W)[:,None,:].expand_as(y1)
    data_loss=(LOCATION_A*(prediction-effective).square()+(LOCATION_D-LOCATION_C**2/LOCATION_A)*y0.square()).mean()
    expected_loss=LOCATION_A*(effective.square().mean()-singular[0].square()/y0.shape[2])+(LOCATION_D-LOCATION_C**2/LOCATION_A)*y0.square().mean()
    transformed=L.T@W
    residual=(transformed-T).square().sum()
    optimal_residual=singular[1:].square().sum()
    grad_beta=2*LOCATION_A/y0.shape[2]*(G@beta*direction.square().sum()-H@direction)
    grad_direction=2*LOCATION_A/y0.shape[2]*(direction*(beta@(G@beta))-H.T@beta)
    runtime_beta=beta.float().double();runtime_direction=direction.float().double()
    second=singular[1] if len(singular)>1 else singular.new_zeros(())
    gap=singular[0]-second
    report={'learner':'public_cosine_rank1_global_fit','degree':DEGREE,'n_features':DEGREE+1,'lambda':0.0,
        'n_train_endpoints':len(x),'N':y0.shape[1],'d':y0.shape[2],
        'gram_condition':condition,'gram_min_eigenvalue':eig[0].item(),
        'singular_values':singular.tolist(),'top_singular_gap':gap.item(),
        'top_singular_numerical_tie':bool(gap<=1e-10*singular[0]),
        'direction_identified_numerically':bool(singular[0]>0 and gap>1e-10*singular[0]),
        'direction_sign_pivot':pivot,'direction_norm':direction.norm().item(),
        'data_loss':data_loss.item(),'objective':data_loss.item(),'penalty':0.0,
        'complete_loss_from_optimality_identity':expected_loss.item(),
        'complete_loss_identity_absolute_error':abs(data_loss.item()-expected_loss.item()),
        'parameter_dependent_excess':(-LOCATION_A*singular[0].square()/y0.shape[2]).item(),
        'whitened_rank_one_residual':residual.item(),'sum_remaining_singular_values_squared':optimal_residual.item(),
        'rank_one_residual_identity_error':abs(residual.item()-optimal_residual.item()),
        'beta_gradient_norm':grad_beta.norm().item(),'direction_gradient_norm':grad_direction.norm().item(),
        'runtime_coefficient_max_abs_error':(runtime_beta[:,None]*runtime_direction[None,:]-W).abs().max().item(),
        'runtime_input_matched_prediction_max_abs_error':(((cosine_features(x.float())@beta.float())[:,None]*direction.float()[None,:]).double()-(cosine_features(x.float().double())@beta)[:,None]*direction[None,:]).abs().max().item(),
        'interpretation':'Single global rank-one fit (tie status recorded), not validation-selected; finite public feature learner differs from tanh NN. Full time-integrated observed loss includes token variance and constants.'}
    if not math.isfinite(report['data_loss']) or not torch.isfinite(W).all():raise FloatingPointError('Nonfinite global fit')
    return field,report,{'G':G,'H':H,'L':L,'T':T,'W':W,'beta':beta,'direction':direction,'singular_values':singular}
