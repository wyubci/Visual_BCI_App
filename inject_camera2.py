import codecs

file_path = r'interface/car_interface/car_window.py'
with codecs.open(file_path, 'r', 'utf-8') as f:
    text = f.read()

# Insert the camera methods right before `def setQss(self):`
if 'def _initCamera(self):' not in text:
    methods_new = '''    def _initCamera(self):
        # 初始化Astra摄像头(尝试多种挂载方式)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.camera_label.setText("未检测到Astra摄像头")
                return

        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera_frame)
        self.camera_timer.start(30)

    def update_camera_frame(self):
        if self.camera_label.isHidden():
            return
            
        ret, frame = self.cap.read()
        if ret:
            # 翻转镜像并转换颜色空间 BGR -> RGB
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio)
            self.camera_label.setPixmap(pixmap)
            
    def closeEvent(self, event):
        if hasattr(self, 'camera_timer') and self.camera_timer.isActive():
            self.camera_timer.stop()
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        try:
            super().closeEvent(event)
        except:
            pass
            
    def setQss(self):'''
    text = text.replace('    def setQss(self):', methods_new)

# Connect init call
if 'self._initCamera()' not in text:
    init_loc = '''        self._initLayout()
        self._initItems()'''
    init_new = '''        self._initLayout()
        self._initItems()
        self._initCamera()'''
    text = text.replace(init_loc, init_new)

# Insert layout part
import re
layout_pattern = r'(self\.mainVLayout\.addLayout\(self\.row1Layout\)\s*\n\s*# 第二行布局)'
camera_layout = '''
        # 中间监控布局 - Astra 摄像头
        self.cameraLayout = QHBoxLayout()
        self.cameraLayout.setContentsMargins(0, 0, 0, 0)
        self.cameraLayout.addStretch()
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid #009688; background-color: #121212; color: #009688;")
        self.camera_label.setText("正在加载 Astra 摄像头...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFont(QFont("Microsoft YaHei", 16))
        self.cameraLayout.addWidget(self.camera_label)
        self.cameraLayout.addStretch()
        self.mainVLayout.addLayout(self.cameraLayout)
        
        # 第二行布局'''

# Change spacing
text = text.replace('self.mainVLayout.setSpacing(180)', 'self.mainVLayout.setSpacing(30)')
text = re.sub(layout_pattern, camera_layout, text)

with codecs.open(file_path, 'w', 'utf-8') as f:
    f.write(text)

print("success")
