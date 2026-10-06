# 正式实验复现

本目录保留论文使用的五组正式数据、四个神经网络实验和端点 CF 实验代码。
`data/` 的 15 个 CSV 和 5 个配置文件均保持原始字节；所有样本预算、重复、
重启及不利结果都保留。`plot_figures.py` 只重绘论文当前引用的六幅 PDF，
`verify_results.py` 只校验数据、统计量与交付代码，不进行训练。

## 校验与论文绘图

以下命令在论文目录（`experiments/` 的上一级）执行。使用独立 Python 环境，
仅校验和绘图需要 `requirements-figures.txt` 中的 NumPy 与 Matplotlib。

```sh
python3 -m pip install -r experiments/requirements-figures.txt
python3 -B experiments/verify_results.py
python3 -B experiments/plot_figures.py
```

默认图目录为 `../figures/`（相对于本实验目录），输出
`fig_particle_fixed_time.pdf`、`fig_path_matched_ode.pdf`、
`fig_dimension_sensitivity.pdf`、`fig_condition_ablation.pdf`、
`fig_cf_endpoint_terminal.pdf`、`fig_cf_euler_discretization.pdf`。
该命令会更新这六幅 PDF；不生成 PNG、宽表或衍生 CSV。
如需检查重绘而不更新论文图片，可指定另一个目录：

```sh
python3 -B experiments/plot_figures.py --output-dir /tmp/ode_paper_figures_new
```

校验先检查 `provenance/source_manifest.json` 中全部 20 个原始 SHA256，
再核对每个最终轮次的重启选择、全部均值与样本标准差（`ddof=1`）、
CF 完整种子/预算/方法网格、场/端点关系以及两个 Euler 误差分解。
CF 原始摘录未包含重新计算逐端点最大残差所需的全部向量，故校验该已记录
残差摘要，同时从 CSV 独立计算每个种子的聚合恒等式。

## 数据、指标和正式设置

| 数据目录 | 正式设计与原始行数 | 论文指标及归一化 |
|---|---|---|
| `fixed_time` | M=500,1000,2000,4000,8000,16000；7 次重复、2 次重启；42 条选择行/84 条全部重启行 | `test_mse`：无噪声场平方 Frobenius 误差除以 N 和测试样本数 |
| `path_matched` | M=500,1000,2000,4000,8000,16000,32000；5 次重复、4 次重启；35/140 行 | `coupled_endpoint_mse`：学习/参考 Euler-200 端点误差除以 N*d 和评估轨迹数 |
| `dimension_sensitivity` | d=4,8,16,32,64；M=2000,4000,8000,16000；5 次重复、4 次重启；100/400 行 | `test_mse`：无噪声场误差除以 N 和测试样本数 |
| `condition_ablation` | M=8000；4 种模型、5 次重复、4 次重启；20/80 行 | `test_mse`：无噪声场误差除以 N 和测试样本数 |
| `cf_endpoint` | M=500,1000,2000,4000,8000；10 个基础种子；150 条方法结果、200 条 Euler 结果 | `terminal_per_token`、`field_per_token`：平方 Frobenius 误差除以 N 和测试端点数；`euler.csv` 同为每 token 单位 |

神经网络的最终轮次验证误差选择重启，不选择最佳训练轮次。固定时刻与维度
实验均为 2000 epochs，路径与消融为 1500 epochs；batch=1024，学习率=0.001。
路径实验 weight decay=0，其他三组为 1e-6；N=8、q=4、rank=2、hidden=128、
4 个条件，`save_checkpoints=false`。固定时刻/维度/消融使用 30000 个无噪声
场测试样本；路径使用 8192 条轨迹，路径采样和终端积分均为 200 步。
其余正式参数（包括噪声、drift scale 和评估指标）完整保存在各 `config.json`。

种子规则与代码保持一致，repeat/restart 从 0 开始：

- 固定时刻：truth=123，data=1000000+100000*repeat+M，
  init=2000000+10000000*repeat+10000*M+restart，test=3000000。
- 路径：truth=123，data=11000000+100000*repeat+M，
  init=12000000+10000000*repeat+10*M+restart，eval=13000000。
- 维度：truth=123+d，data=1000000+10000000*d+100000*repeat+M，
  init=2000000+10000000*d+100000*repeat+10*M+restart，test=3000000+d。
- 消融：truth=123，data=1000000+100000*repeat+8000，
  init=2000000+10000000*repeat+100000*model_index+restart，test=3000000。
  模型顺序为 proposed、full_rank、no_condition、independent_token。

两幅 particle-field 图使用 beta=2 的 M^(-4/5) 参考线。固定时刻参考线锚定本组
M=16000 的均值，维度参考线独立锚定本组 d=8、M=16000 的均值。
固定时刻全部六个预算均值的无权重 log-log 最小二乘斜率为
-0.8096480761970609（论文写为 -0.810）；这是有限范围的描述性拟合。

## 重新运行正式神经网络实验

从论文目录执行。runner 检查原始配置的 SHA256，只替换新输出位置，
可选 `--device cpu/cuda/auto` 仅改变运行设备；未指定时保持配置记录。

```sh
python3 -m pip install -r experiments/code/neural/requirements.txt
python3 -B experiments/code/run_neural.py fixed_time --output-dir /tmp/ode_formal_new/fixed_time
python3 -B experiments/code/run_neural.py path_matched --output-dir /tmp/ode_formal_new/path_matched
python3 -B experiments/code/run_neural.py dimension_sensitivity --output-dir /tmp/ode_formal_new/dimension
python3 -B experiments/code/run_neural.py condition_ablation --output-dir /tmp/ode_formal_new/ablation
```

每次使用尚不存在、位于整个论文目录外的输出目录。路径配置要求 CUDA；如改用 CPU，需显式加
`--device cpu`。原始四个入口与其库逐字节保留，也会在新运行目录生成原始
辅助汇总与图；当前论文图统一由 `plot_figures.py` 从冻结 CSV 重绘。

## 重新运行正式端点 CF 实验

CF 入口需要 POSIX macOS/Linux。下列命令进入 `code/cf_endpoint/` 后执行，
run 必须是其 `outputs/` 下的全新子目录；不允许参数或种子筛选。

```sh
cd experiments/code/cf_endpoint
python3 -m pip install -r requirements.txt
python3 -B -m endpoint_ode.cf_sweep freeze --run outputs/cf_formal_new --config configs/cf_sweep_formal.json
python3 -B -m endpoint_ode.cf_sweep fit --run outputs/cf_formal_new
python3 -B -m endpoint_ode.cf_sweep evaluate --run outputs/cf_formal_new
python3 -B -m endpoint_ode.cf_sweep audit --run outputs/cf_formal_new
```

`freeze` 不抽取端点；`fit` 生成独立的训练/验证样本池并提交全部解；
`evaluate` 在完整训练屏障通过后生成独立测试端点，保存逐端点向量和 CSV；
`audit` 独立验证 QR/SVD 最优性、完整观测目标、积分、精度、单位和 Euler 分解。
中断的 fit 可用 `resume` 命令继续；已提交阶段不可覆盖，已完成阶段的
验证式续跑可用 `--resume`。新包代码库存已改变，不能续跑旧归档的 run。

正式配置 `purpose=formal` 与数据配置逐字节相同：N=d=8、13 个 cosine 特征、
rank=1、lambda=0，固定真值 J=3、beta=2、delta=1、omega=(1,0,1)、u=e1。
10 个基础种子为 2026100301–2026100310；各随机流由原始
`derive_seed(base_seed, namespace, stream)` SHA256 规则生成。
各预算使用独立训练/验证池的前缀（90%/10%），同一基础种子的全部预算和
方法共享独立的 2048 个测试端点。预算、方法和 token 不是新增独立重复。
拟合使用 CPU float64，运行参数为 float32；解析评分将保存的参数提升至
double，并使用 float32 舍入的 x/z0。线程数为 4，开启确定性算法。
Euler 固定 M=4000，K=10,20,40,80,160；不按目标误差选择 K。

CF 的训练均值是仅由拟合 Y1 估计的无条件基线。最小三个预算上 CF 平均
误差高于两种基线；CF 对 train mean 的配对胜出次数依次为 0、0、2、10、10。
零场 Euler 离散误差为 0，CSV 保留该结果，绘图不将其平移到对数轴。
场/端点聚合关系为 field=(pi^2/4)*terminal；Euler 目标误差中的交叉项也保留。

## 来源与复现范围

`provenance/source_manifest.json` 是原始数据的冻结校验依据；
`neural_source_provenance.json`、`cf_source_provenance.json` 与
`historical_environment.json` 均原样保留，是历史来源/环境记录，
其中旧 copied_file 标识不代表本次交付文件清单。
`provenance/delivery_manifest.json` 是清理完成后新生成的交付校验清单，
记录当前路径与哈希；它不是原始运行前的代码承诺或签名。

此次只清理包装与未使用的诊断接口：删除 smoke runner/config/tests 和 CF
可选诊断 plot，更新 CF 冻结代码库存；正式设计、核心求解与评分不变。
原始 20 个数据文件、四个神经入口/库、CF cosine 核心及其他科学库字节不变。
未重新运行完整训练，也不声称按位复现。神经实验历史日志记录 CUDA/RTX 5090，
缺少关联到这四次运行的精确 Python/PyTorch/CUDA 版本及完整搜索历史/检查点；
CF 计划记录 Python 3.12.2、PyTorch 2.8.0、NumPy 2.3.3、SciPy 1.16.2。
神经 requirements 为原始最低版本要求，CF requirements 为原始计划固定版本。
