# CAD 自动装配环境配置

## 已验证环境

本工程于 2026-07-01 在 Windows 11、64 位 Python 3.10 和
conda-forge `pythonocc-core` 环境完成基线复现。几何主流程只使用 CPU，
没有 GPU 也应能运行。

当前机器已安装用户级 Miniforge，环境位置为
`C:\Users\11049\miniforge3\envs\cad_asm`。首次打开新终端后，可执行：

```powershell
conda activate cad_asm
cd C:\Users\11049\Desktop\Model_match\sw
python check_env.py .
```

如果当前终端尚未识别 `conda`，使用 `conda run`：

```powershell
& "$HOME\miniforge3\Scripts\conda.exe" run --no-capture-output `
  -n cad_asm python check_env.py .
```

不要直接执行
`C:\Users\11049\miniforge3\envs\cad_asm\python.exe`。Windows 下这种启动
方式不会可靠注入 Conda 的 `Library\bin`；NumPy 可能可以导入，却在
首次调用 SVD/LAPACK 时因延迟加载 DLL 失败而原生退出。

## 从零创建环境

推荐 Windows 10/11、Miniforge/Miniconda 和 Python 3.10。`pythonocc-core`
必须从 conda-forge 安装，不要依赖系统 Python 3.12 的 `pip`：

```powershell
conda create -n cad_asm -c conda-forge `
  python=3.10 pythonocc-core numpy scipy pandas networkx pywin32 `
  tqdm pydantic matplotlib scikit-learn -y
conda activate cad_asm
```

第一轮基线不需要 OR-Tools、PuLP、XGBoost、LightGBM、PyTorch 或 CUDA。
后续若实现可靠搜索器或学习型 match scorer，再按模块单独添加。

## 自检

```powershell
python check_env.py .
python check_env.py . --output environment_report.json
python check_env.py . --skip-solidworks
```

脚本检查 Python、OCC、数值库、实际 SVD/LAPACK 运算、pywin32、
SolidWorks COM、Torch/CUDA、NVIDIA 驱动、工作目录读写权限和 STEP
样例。数值探针在子进程运行，因此 DLL 原生崩溃也会被记录到 JSON。

## 原始命令

在 `sw` 目录中运行：

```powershell
python compute_manifest.py .\1
python compute_manifest.py .\1 --write-diagnostics
python build_assembly.py .\1

python compute_manifest.py .\4 --decompose
python build_assembly.py .\4 --use-parents
```

批量基线：

```powershell
python run_cases.py . --solver bfs
python run_cases.py . --solver bfs --cases 1 2
python run_cases.py . --solver bfs --cases 3 4 5 --compute-only
```

结果写入 `baseline_results/summary.csv`、`baseline_results/logs/` 和
`baseline_results/reports/`。批跑会执行原命令并覆盖 case 中同名的
`assembly_manifest.json` 与 `assembly.step`。

### 本机稳定性限制

2026-07-01 的复现中，OCCT 处理大 STEP 时出现用户态访问违规和一次系统
BugCheck `0xF7`。在 BIOS/硬件稳定性检查完成前，只运行 case 1/2；对
case 3/4/5 使用 `--compute-only`，不要执行 STEP 合并导出。事故证据和
WinDbg 结论见 `../INCIDENT_REPORT.md`。

## GPU 与 NVIDIA 驱动

`nvidia-smi` 只验证系统 NVIDIA 驱动及其支持的最高 CUDA runtime。
它不表示 PyTorch 已安装，也不等于 PyTorch 使用的 CUDA 版本。代码必须
通过 `torch.cuda.is_available()` 决定使用 `cuda` 还是 `cpu`。

当前机器能识别 NVIDIA GPU，但基线环境没有安装 PyTorch，因此设备报告
为 CPU；这不影响 STEP 解析、特征提取、匹配、BFS 或 STEP 导出。

## SolidWorks API

程序化生成 `.sldprt/.sldasm/.step` 需要本机安装并完成首次启动初始化的
SolidWorks。COM 最小验证代码为：

```python
import win32com.client
sw_app = win32com.client.Dispatch("SldWorks.Application")
print(sw_app.RevisionNumber())
```

如果失败，依次检查 SolidWorks 是否安装、当前用户是否可启动它、Python
和 COM 是否同为 64 位、`pywin32` 是否安装，以及 SolidWorks 是否完成
首次启动。当前机器已验证 SolidWorks COM 可连接，版本为 `34.2.1`
（SOLIDWORKS 2026 SP2.1）。
