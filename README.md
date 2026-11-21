# OSOT 播音分谱工作台与协议编译器

用于查看和编辑播音表达标注，并将标注数据整理为面向后续语音研究的分谱格式。

## 快速体验

在浏览器打开 `osot_v02_workbench.html`，或用 Python 运行：

```bash
python osot_score_compiler.py --demo --out-dir demo_compile_out
python osot_score_compiler.py --json 人工终稿.json --auto-json 机器初标.json --out-dir compiled_out
```

Windows 用户也可运行 `run_compiler.bat`。仓库保留原版、v2 脚本和共用数据协议；启动脚本使用原版。
