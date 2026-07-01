from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qfluentwidgets import setTheme, Theme, setThemeColor
from utils.logger import logger
import sys
import os
import ctypes

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from config.configer import config

def my_excepthook(exc_type, exc_value, tb):
    msg = ' Traceback (most recent call last):\n'
    while tb:
        filename = tb.tb_frame.f_code.co_filename
        name = tb.tb_frame.f_code.co_name
        lineno = tb.tb_lineno
        msg += ' File "%.500s", line %d, in %.500s\n' % (filename, lineno, name)
        tb = tb.tb_next
        msg += ' %s: %s\n' %(exc_type.__name__, exc_value)

    print(msg)
    try:
        box = QMessageBox()
        box.setWindowFlags(box.windowFlags() | Qt.WindowStaysOnTopHint)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle('错误')
        box.setText('请联系工程师， 查看日志')
        box.exec_()
    except Exception:
        pass
    logger.warning(msg)

if __name__ == '__main__':
    # 不再强制覆盖被试名，避免每次启动都把数据写入 hc33。
    # 如需更换被试，可在 config.yaml 中修改 subjectName，或在用户界面中扩展被试选择。
    if not getattr(config, 'subjectName', None):
        config.subjectName = 'TestSubject'
        config.save()
    print(f'start subject={config.subjectName}')

    # 处理任务栏图标不显示问题
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("analysis")

    sys.excepthook = my_excepthook
    setTheme(Theme.DARK)
    setThemeColor(QColor(Qt.white))
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication([])
    from interface.home_interface.home_window import HomeWindow

    main = HomeWindow()
    main.showMaximized()
    app.exec_()

    os._exit(0)
