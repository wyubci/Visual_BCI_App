import ctypes
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import QApplication, QMessageBox, QVBoxLayout, QWidget
from qfluentwidgets import DisplayLabel, ProgressBar, Theme, setTheme, setThemeColor

from interface.car_interface.car_video_panel import CarVideoPanel
from utils.logger import logger


class CarVideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('脑控小车视频监看')
        self.resize(760, 620)
        self.setObjectName('carVideoWindow')
        self._initLayout()

    def _initLayout(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(12)

        self.title_label = DisplayLabel()
        self.title_label.setText('脑控小车视频监看')
        self.title_label.setFixedHeight(72)
        font = QFont()
        font.setPixelSize(24)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)

        self.status_bar = ProgressBar()
        self.status_bar.setRange(0, 0)
        layout.addWidget(self.status_bar)
        self.video_panel = CarVideoPanel(width=640, height=480, parent=self)
        layout.addWidget(self.video_panel, 1)

    def closeEvent(self, event):
        if hasattr(self, 'video_panel') and self.video_panel is not None:
            self.video_panel.close()
        super().closeEvent(event)


def my_excepthook(exc_type, exc_value, tb):
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
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('analysis.car.video')

    sys.excepthook = my_excepthook
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)

    setTheme(Theme.DARK)
    setThemeColor(QColor(Qt.white))

    window = CarVideoWindow()
    window.show()
    sys.exit(app.exec_())