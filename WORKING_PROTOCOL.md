# Codex 稳定工作约定

为避免 VS Code/Codex Webview 再次灰屏，本项目采用以下执行约定：

1. 前台工具调用尽量控制在 60 秒内。
2. 长时间 CAD、WinDbg 或批处理任务使用隐藏后台进程。
3. 后台任务写入独立日志和 PID 文件，并按阶段汇报状态。
4. 不向 Codex 面板回传超大安装日志、实体列表或调试器全文。
5. 大 STEP 按 case 单独运行，不并行占用 OCC 原生库。
6. 每个 case 开始前备份或隔离旧 manifest，完成后立即验证文本文件。
7. case 3/4/5 在确认系统稳定前，先 compute-only，再单独决定是否导出。
8. 任何超过一分钟的操作都要向用户说明当前阶段和下一检查点。

UI 稳定配置：

- VS Code Electron 硬件加速已关闭（下次重启后生效）。
- 第三方 DeepSeek Copilot 扩展已卸载，减少 extension host 冲突。
- Codex/OpenAI 扩展保留启用。
# Task completion marker

- At the start of a new long task, remove the previous
  `C:\Users\11049\Desktop\Codex.txt` marker.
- Create `C:\Users\11049\Desktop\Codex.txt` with the exact content
  `工作完成` only after the requested task has passed its completion checks.
- For the current task, continue through D4 and D5, then stop before the first
  DeepSeek API call and ask the user for `DEEPSEEK_API_KEY`.
- Do not treat a background process, partial result, or API blocker as task
  completion.
