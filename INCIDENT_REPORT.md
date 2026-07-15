# 2026-07-01 CAD 基线复现异常报告

## 结论

本次不是普通界面卡死。Windows 于 2026-07-01 14:43 左右发生
BugCheck `0xF7 DRIVER_OVERRAN_STACK_BUFFER`，14:49 重启。内核 dump
显示崩溃上下文为 `python.exe`，系统执行内核栈回溯时发现 security
cookie 被覆盖：

```text
FAILURE_BUCKET_ID: 0xF7_MISSING_GSFRAME_nt!_report_gsfailure
PROCESS_NAME: python.exe
SECURITY_COOKIE: expected 0000f7eb68e82e7b
                 found    ffffb79d59db1570
```

dump 的调用栈已损坏，无法可靠点名某个第三方内核驱动。它能证明 Python
任务是触发上下文，但不能证明普通 Python 代码能直接破坏内核。

同一阶段的用户态 dump 更明确：OpenCASCADE 7.9.3 在读取/修复大 STEP
时触发 `FAST_FAIL_STACK_COOKIE_CHECK_FAILURE`：

```text
TKernel.dll
TKShHealing!ShapeFix_*
TKDESTEP!STEPControl_ActorRead::Transfer*
```

case 3 后续还在 `VCRUNTIME140.dll` 发生 `0xC0000005`；OCCT 7.7.2
兼容环境也在写出阶段发生访问违规。因此问题不只是某一个 pythonocc
版本。

## 伴随证据

- 14:09 有一条 WHEA “Processor Core / Translation Lookaside Buffer”
  已纠正硬件错误。
- 更早有一次磁盘 paging operation 警告。
- 重启后发现 `check_env.py`、4 个 manifest 和若干 JSON 报告被完整清零。
- SSD SMART/Storage 状态显示 Healthy/OK。
- 内核 dump 的 NTFS 黑盒为 0 条慢 I/O、0 条 oplock 超时。
- PNP 黑盒仅记录音频端点 problem code 24，没有指向 CAD 任务。
- 机器为 ASUS GU604VI，当前 BIOS 313；华硕已发布并推荐 BIOS 314。

综合判断：大 STEP 触发的 OCCT 原生栈损坏是最直接的软件触发器；BIOS、
CPU/内存或系统运行时不稳定可能是放大因素。现有证据不足以单独归因。

## 已采取措施

1. 停止 case 3/4/5 的 STEP 合并导出。
2. `run_cases.py` 增加 `--cases` 和 `--compute-only` 安全开关。
3. `check_env.py` 增加独立子进程 SVD/LAPACK 实算检查。
4. case 1/2 已在正确激活的环境下重新运行并成功。
5. 保留内核 dump、WinDbg 输出和命令文件于 `incident_dumps/`。
6. 扫描源码目录中的空字节，并重建损坏的 `check_env.py`。

## 恢复前建议

以下操作需要管理员权限或重启，尚未自动执行：

1. 接通电源，暂停 BitLocker 后按华硕官方流程将 BIOS 313 更新到 314。
2. 安装 GU604VI 官方 SSD firmware update。
3. 以管理员终端运行 `chkdsk C: /scan` 和 `sfc /scannow`。
4. 运行 Windows Memory Diagnostic；若再次出现 WHEA，使用 MemTest86
   做更长时间测试。
5. 暂时关闭超频、降压和激进性能配置，使用厂商默认电压/频率。
6. 完成上述检查后，先在复制出的测试目录单独运行 case 3，再测试
   case 4；case 5 最后运行。

固件升级具有断电和不可启动风险，不应在无人确认供电、BitLocker 和恢复
密钥状态时自动执行。
