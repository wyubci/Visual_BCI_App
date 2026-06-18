from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qfluentwidgets import setTheme, Theme, setThemeColor
from utils.logger import logger
import sys
import os
import ctypes
from config.configer import config

def my_excepthook(exc_type, exc_value, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, tb)
        return

    msg = ' Traceback (most recent call last):\n'
    while tb:
        filename = tb.tb_frame.f_code.co_filename
        name = tb.tb_frame.f_code.co_name
        lineno = tb.tb_lineno
        msg += ' File "%.500s", line %d, in %.500s\n' % (filename, lineno, name)
        tb = tb.tb_next
    msg += ' %s: %s\n' % (exc_type.__name__, exc_value)

    print(msg)
    app = QApplication.instance()
    if app is not None:
        msg_box = QMessageBox(QMessageBox.Warning, '错误', '请联系工程师，查看日志')
        msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
        msg_box.exec_()
    logger.error(msg)

if __name__ == '__main__':
    # 确保只运行一个实例
    app = QApplication.instance()
    if app is None:
        try:
             # 启用高DPI缩放
            QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
            app = QApplication(sys.argv)
        except Exception as e:
            print(f"QApplication init failed: {e}")
            app = QApplication(sys.argv)
    else:
        print("Application already exists, using existing instance.")

    # 请输入被试姓名日期_批次_刺激时间
    config.subjectName = 'hc33'
    config.save()
    print('start')

    # 处理任务栏图标不显示问题
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("analysis")
    except: pass

    sys.excepthook = my_excepthook
    
    try:
        setTheme(Theme.DARK)
        setThemeColor(QColor(Qt.white))
    except Exception as e:
        print(f"Theme setting failed: {e}")

    try:
        from interface.home_interface.home_window import HomeWindow
        main = HomeWindow()
        main.showMaximized()
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Main loop error: {e}")
        # 如果是重启模式，不要直接退出
        if "restart" not in sys.argv:
            sys.exit(1)

