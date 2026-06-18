# Windows 中文显示说明

本项目的关键源码和说明文件已经统一保存为 `UTF-8 with BOM`，Windows 记事本、Word、VS Code 和 PyCharm 都可以正常识别中文。

如果在 PowerShell 里使用 `Get-Content` 仍然看到乱码，通常是控制台代码页的问题。可以先执行：

```powershell
chcp 65001
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

推荐使用 VS Code 打开文件，并在右下角确认编码为 `UTF-8 with BOM`。

有时终端日志显示乱码，但 Python 实际读取文件仍然是正确中文。可以运行下面的命令验证：

```powershell
F:\anaconda\envs\bci_env\python.exe -c "from pathlib import Path; p=Path('interface/car_interface/car_window.py'); t=p.read_text(encoding='utf-8-sig'); i=t.find('self.commands'); print(t[i:i+80])"
```

正常输出应包含：

```text
self.commands = [
    "前进", "后退", "左转", "停止", "右转"
]
```

