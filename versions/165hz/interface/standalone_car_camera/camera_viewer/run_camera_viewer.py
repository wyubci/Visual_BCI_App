import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QFont

from car_video_panel import CarVideoPanel


class CameraViewerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Standalone Car Camera Viewer")
        self.resize(760, 620)

        layout = QVBoxLayout(self)
        title = QLabel("Standalone Car Camera Viewer")
        title.setFont(QFont("Segoe UI", 16))
        layout.addWidget(title)

        self.video_panel = CarVideoPanel(width=640, height=480, parent=self)
        layout.addWidget(self.video_panel)

    def closeEvent(self, event):
        if hasattr(self, "video_panel") and self.video_panel is not None:
            self.video_panel.close()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CameraViewerWindow()
    w.show()
    sys.exit(app.exec_())
