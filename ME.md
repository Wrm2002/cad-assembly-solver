# Codex working conventions

## Long-running work

- Do not hold the Codex UI on a long foreground experiment.
- Run long experiments through a hidden background process with stdout,
  stderr, PID, heartbeat/progress, and result files.
- Check progress in short calls and keep the user informed.
- Prefer bounded smoke tests before full-dataset runs to reduce system pressure.
