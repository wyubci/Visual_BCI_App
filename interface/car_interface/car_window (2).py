import socket
import pyttsx3
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qfluentwidgets import *
import threading
import numpy as np
import time
from scipy.io import savemat, loadmat
from glob import glob
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.TDCA import TDCA_Adapter
from config import config
from interface.car_interface.car_video_panel import CarVideoPanel

# ==========================================
# PC CLIENT FOR X1
# Run this on Windows
# ==========================================

ROBOT_IP = '192.168.1.11'  # CHANGE THIS IF NEEDED
PORT = 65432

class RobotClient:
    def __init__(self, ip, port):
        self.addr = (ip, port)
        self.sock = None
        self.connected = False

    def connect(self):
        if self.connected:
            return
        try:
            print(f"Connecting to {self.addr}...")
            if self.sock:
                try: self.sock.close()
                except: pass
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect(self.addr)
            self.connected = True
            print("Connected!")
        except Exception as e:
            print(f"Connection Failed: {e}")
            self.connected = False

    def send(self, cmd):
        if not self.connected:
            self.connect()
            
        if not self.connected: return
        
        try:
            self.sock.send(cmd.encode('utf-8'))
        except Exception as e:
            print(f"Send Error: {e}, Retrying...")
            self.connected = False
            self.connect()
            if self.connected:
                try:
                    self.sock.send(cmd.encode('utf-8'))
                except:
                    self.connected = False

    def set_motor(self, v1, v2, v3, v4):
        cmd = f"{int(v1)},{int(v2)},{int(v3)},{int(v4)}"
        print(f"Sending: {cmd}")
        self.send(cmd)

    def move(self,ing):
        if(ing==0):
            v1=30
            v2=30
            v3=30
            v4=30
            self.set_motor(v1, v2, v3, v4)
            time.sleep(4)
            self.set_motor(0,0,0,0)
        elif(ing==1):
            v1=-30
            v2=-30
            v3=-30
            v4=-30
            self.set_motor(v1, v2, v3, v4)
            time.sleep(4)
            self.set_motor(0,0,0,0)
        elif(ing==2):
            v1=-65
            v2=0
            v3=65
            v4=0
            self.set_motor(v1, v2, v3, v4)
            time.sleep(4)
            self.set_motor(0,0,0,0)
        elif(ing==3):
            v1=0
            v2=0
            v3=0
            v4=0
            self.set_motor(v1, v2, v3, v4)

        elif(ing==4):
            v1=65
            v2=0
            v3=-65
            v4=0
            self.set_motor(v1, v2, v3, v4)
            time.sleep(4)
            self.set_motor(0,0,0,0)
        
    def stop(self):
        print("Stopping")
        self.send("stop")

    def beep(self):
        print("Beep")
        self.send("beep")

    def close(self):
        self.sock.close()

# 小车移动每秒5cm，旋转每秒18度
class StiRect(QLabel):
    def __init__(self, text, parent, sti, fontSize=60, color=255):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.fontSize = fontSize
        self.sti = sti
        self.text = text
        self.color_value = color
        self.current_color = QColor(color, color, color, 255)
        self.default_color = QColor(color, color, color, 255)

    def changeText(self, text):
        self.text = text
        self.update()

    def flicker(self, freq, now_time):
        light = 255 * np.sin(2 * np.pi * now_time * freq)
        return int(light)

    def changeColor(self, f, now_time):
        color = self.flicker(f, now_time)
        self.color_value = color
        self.current_color = QColor(255, 255, 255, color)
        self.update()

    def setDefaultColor(self):
        self.current_color = self.default_color
        self.color_value = 255
        self.update()

    def paintEvent(self, a0):
        super().paintEvent(a0)

        painter = QPainter()
        painter.begin(self)
        painter.setRenderHints(QPainter.Antialiasing)
        
        # 绘制发光/阴影效果 (Cyberpunk style)
        if self.current_color.red() < 255 or self.current_color.green() < 255 or self.current_color.blue() < 255:
            # 当从白色变化到灰度/黑色时（闪烁状态），稍微改变背景绘制
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.current_color)
            painter.drawRoundedRect(self.rect().adjusted(4, 4, -4, -4), 10, 10)
        else:
            # 默认状态 (白色) - 使用半透明填充 + 亮色边框 + 阴影模拟
            painter.setPen(QPen(QColor(0, 255, 255), 2)) # 青色边框
            painter.setBrush(QColor(255, 255, 255, 220)) # 略微透明的白色
            painter.drawRoundedRect(self.rect().adjusted(4, 4, -4, -4), 10, 10)

        text_color = QColor(0, 0, 0)
        if self.current_color.red() < 100: # 如果背景很暗，字变白
             text_color = QColor(0, 255, 255) # 青色字

        font = QFont("Microsoft YaHei")
        font.setPixelSize(self.fontSize)
        font.setBold(True)

        pen = QPen()
        pen.setColor(text_color)
        
        painter.setPen(pen)
        painter.setFont(font)
        
        # 绘制文字
        painter.drawText(self.rect().adjusted(1, 1, -1, -1), Qt.AlignCenter, self.text)
        
        # 绘制频率角标
        freq_font = QFont("Arial")
        freq_font.setPixelSize(14)
        painter.setFont(freq_font)
        painter.setPen(QPen(QColor(100, 100, 100) if text_color.red() == 0 else QColor(0, 200, 200)))
        painter.drawText(self.rect().adjusted(10, 10, -10, -10), Qt.AlignTop | Qt.AlignRight, f"{self.sti}Hz")

        painter.end()

class CarControlWindow(QWidget):
    update_text = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    
    def __init__(self, objectName):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setObjectName(objectName)
        
        # Connect signals
        self.update_text.connect(self.safe_set_text)
        self.update_progress.connect(self.safe_set_progress)
        
        self.setQss()
        
        # 设置刺激频率列表 - 5个频率
        self.sti_lst = [
            8, 9, 9.5, 10, 10.5
        ]
        
        # 命令列表：前进、后退、左转、停止、右转
        self.commands = [
            "前进", "后退", "左转", "停止", "右转"
        ]
        
        self.cache_data = np.array([])
        self.start_flick = False
        self.finish = True
        self.start_cache = False
        self.continuous_mode = False  # 连续刺激模式
        current_time_str = time.strftime('%Y%m%d_%H%M%S')
        self.exp_result_dir = os.path.join('ExperimentResults', current_time_str)
        if not os.path.exists(self.exp_result_dir):
            os.makedirs(self.exp_result_dir)
        self.exp_txt_path = os.path.join(self.exp_result_dir, 'results.txt')
        try:
            with open(self.exp_txt_path, 'a', encoding='utf-8') as f: f.write(f'--- 脑控小车实验记录 ({current_time_str}) ---\n')
        except Exception: pass
        
        self._initLayout()
        self._initItems()

        self.times = 0.5
        self.fbcca = TDCA_Adapter(3, self.times, self.sti_lst)
        
        # 训练数据缓存 (用于累积训练)
        self.history_X_list = []
        self.history_y_list = []
        self.load_history()
        
        # 初始化持久化的小车客户端
        self.robot_client = RobotClient(ROBOT_IP, PORT)
        # 尝试首次连接(非阻塞，失败也没关系，发送时会自动重连)
        threading.Thread(target=self.robot_client.connect).start()

        # 设置焦点策略，使窗口可以接收键盘事件
        self.setFocusPolicy(Qt.StrongFocus)

    def setQss(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        qss_path = os.path.join(base_dir, 'source', 'qss', 'mainWindow.qss')
        try:
            with open(qss_path, encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        except: pass
        
        # 添加极简黑色风格 + 赛博朋克装饰
        self.setStyleSheet(self.styleSheet() + """
            QWidget#carControlWindow { 
                background-color: #050505; 
                background-image: linear-gradient(135deg, #050505 0%, #0a0a0a 100%);
            }
            QLabel {
                color: #e0e0e0;
                font-family: "Microsoft YaHei";
            }
            ProgressBar {
                border: 1px solid #333;
                background-color: #1a1a1a;
                border-radius: 4px;
                text-align: center;
                color: white;
            }
            ProgressBar::chunk {
                background-color: #00E5FF; /* 霓虹蓝 */
                border-radius: 3px;
            }
        """)

    def safe_set_text(self, text):
        if hasattr(self, 'show_label'):
            self.show_label.setText(text)
            
    def safe_set_progress(self, val):
        if hasattr(self, 'progress'):
            self.progress.setValue(val)

    def _initLayout(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 10, 20, 20)
        self.layout.setSpacing(1)
        self.setLayout(self.layout)

    def _initItems(self):
        # 标题标签
        self.show_label = DisplayLabel()
        self.show_label.setText("脑控小车界面")
        self.show_label.setFixedHeight(80)
        font = QFont()
        font.setPixelSize(24)
        self.show_label.setFont(font)
        self.layout.addWidget(self.show_label)

        # 进度条
        self.progress = ProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)
        self.layout.addSpacing(10)

        # 训练按钮
        self.train_btn = QPushButton("开始模型训练 (Calibration)", self)
        self.train_btn.clicked.connect(self.start_calibration)
        self.train_btn.setCursor(Qt.PointingHandCursor)
        self.train_btn.setFixedHeight(40)
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #00E5FF;
                color: black;
                border-radius: 5px;
                font-weight: bold;
                font-family: "Microsoft YaHei";
            }
            QPushButton:hover {
                background-color: #00B8D4;
            }
            QPushButton:pressed {
                background-color: #00838F;
            }
        """)
        self.layout.addWidget(self.train_btn)

        self.layout.addSpacing(20)
        
        # --- 恢复原有布局 (Restored Layout) ---
        self.sti_rects = []
        
        # 创建垂直布局容器
        self.mainVLayout = QVBoxLayout()
        self.mainVLayout.setContentsMargins(50, 0, 50, 50)
        self.mainVLayout.setSpacing(20)
        self.layout.addLayout(self.mainVLayout)
        
        # 第一行布局 - 前进、后退
        self.row1Layout = QHBoxLayout()
        self.row1Layout.setContentsMargins(0, 0, 0, 0)
        self.mainVLayout.addLayout(self.row1Layout)

        # 中间监控布局 - 小车视频 (居中)
        self.cameraLayout = QHBoxLayout()
        self.cameraLayout.setContentsMargins(0, 0, 0, 0)
        self.cameraLayout.addStretch()
        self.video_panel = CarVideoPanel(width=560, height=420, parent=self)
        self.cameraLayout.addWidget(self.video_panel)
        self.cameraLayout.addStretch()
        self.mainVLayout.addLayout(self.cameraLayout)
        
        # 第二行布局 - 左转、停止、右转
        self.row2Layout = QHBoxLayout()
        self.row2Layout.setContentsMargins(0, 0, 0, 0)
        self.mainVLayout.addLayout(self.row2Layout)
        
        rect_size = 180  # 刺激区域大小
        
        # 第一行 - 前进、后退 (尽可能分散)
        self.row1Layout.addStretch(1)
        
        # 前进 (8Hz)
        self.forward_rect = StiRect("前进", self, self.sti_lst[0])
        self.forward_rect.setFixedSize(rect_size, rect_size)
        self.row1Layout.addWidget(self.forward_rect)
        self.sti_rects.append(self.forward_rect)
        
        self.row1Layout.addStretch(3) # 中间大间距
        
        # 后退 (9Hz)
        self.backward_rect = StiRect("后退", self, self.sti_lst[1])
        self.backward_rect.setFixedSize(rect_size, rect_size)
        self.row1Layout.addWidget(self.backward_rect)
        self.sti_rects.append(self.backward_rect)
        
        self.row1Layout.addStretch(1)
        
        # 第二行 - 左转、停止、右转 (均匀分散)
        self.row2Layout.addStretch(1)
        
        # 左转 (9.5Hz)
        self.left_rect = StiRect("左转", self, self.sti_lst[2])
        self.left_rect.setFixedSize(rect_size, rect_size)
        self.row2Layout.addWidget(self.left_rect)
        self.sti_rects.append(self.left_rect)
        
        self.row2Layout.addStretch(2)
        
        # 停止 (10Hz)
        self.stop_rect = StiRect("停止", self, self.sti_lst[3])
        self.stop_rect.setFixedSize(rect_size, rect_size)
        self.row2Layout.addWidget(self.stop_rect)
        self.sti_rects.append(self.stop_rect)
        
        self.row2Layout.addStretch(2)
        
        # 右转 (10.5Hz)
        self.right_rect = StiRect("右转", self, self.sti_lst[4])
        self.right_rect.setFixedSize(rect_size, rect_size)
        self.row2Layout.addWidget(self.right_rect)
        self.sti_rects.append(self.right_rect)

        self.row2Layout.addStretch(1)
        
        self.layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Maximum, QSizePolicy.Expanding))
    
    def setDefaultColor(self):
        for rect in self.sti_rects:
            rect.setDefaultColor()

    def setCue(self, idx):
        # 高亮提示目标
        self.setDefaultColor()
        for i, rect in enumerate(self.sti_rects):
            if i == idx:
                rect.color_value = 0 # 提示颜色 (黑色)
                rect.current_color = QColor(255, 0, 0, 200) # 红色提示
                rect.update()

    def load_history(self):
        calib_dir = os.path.join('saveCarData', config.subjectName, 'calibration')
        print(f"Loading history from: {calib_dir}")
        if not os.path.exists(calib_dir):
            return
            
        mat_files = glob(os.path.join(calib_dir, '*.mat'))
        count = 0
        total_trials = 0
        
        for f in mat_files:
            try:
                data = loadmat(f)
                if 'X' in data and 'y' in data:
                    X = data['X']
                    y = data['y']
                    # X shape: (trials, channels, points)
                    # y shape: (trials,) or (1, trials)
                    
                    if y.ndim > 1:
                        y = y.flatten()
                        
                    self.history_X_list.append(X)
                    self.history_y_list.append(y)
                    count += 1
                    total_trials += len(y)
            except Exception as e:
                print(f"Failed to load {f}: {e}")
                
        if total_trials > 0:
            print(f"已自动加载 {count} 个历史校准文件，共包含 {total_trials} 条训练数据。")
            if count > 0:
                self.fit_model_from_history()
                self.train_btn.setText(f"开始增量训练 ({total_trials}样本)")
        else:
             print("未找到历史训练数据，将使用全新训练模式。")

    def fit_model_from_history(self):
        if len(self.history_X_list) == 0:
            return
        
        try:
            # 寻找全局最小长度 (确保所有trial长度一致)
            global_min_len = min([x.shape[-1] for x in self.history_X_list])
            
            # 统一截断并合并
            final_X_list = [x[..., :global_min_len] for x in self.history_X_list]
            train_X = np.concatenate(final_X_list, axis=0)
            train_y = np.concatenate(self.history_y_list, axis=0)
            
            # 立即训练模型
            self.fbcca.fit(train_X, train_y)
            print(f"模型已使用历史数据完成训练，样本数: {train_X.shape[0]}")
            self.update_text.emit(f"模型已加载 {train_X.shape[0]} 条历史数据并完成训练")
            
        except Exception as e:
            print(f"Fit History Error: {e}")

    def start_calibration(self):
        if self.start_flick:
            return
        
        # 增加提示：是否清空历史数据？
        if hasattr(self, 'history_X_list') and len(self.history_X_list) > 0:
            total_history = sum([len(y) for y in self.history_y_list])
            reply = QMessageBox.question(self, '增量训练', 
                f"检测到 {total_history} 条历史数据。\n是否基于已有数据继续训练 (Yes)，还是清空历史重新开始 (No)？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
            if reply == QMessageBox.No:
                self.history_X_list = []
                self.history_y_list = []
                self.fbcca.is_fitted = False
                self.train_btn.setText("开始模型训练 (Calibration)")
                
        self.train_btn.setEnabled(False)
        self.update_text.emit("准备开始训练...")
        
        # 启动训练线程
        self.finish = False
        threading.Thread(target=self.calibration_task).start()

    def calibration_task(self):
        # 训练参数
        trials = 3  # 提高为3轮，增加数据多样性
        calibration_time = 1.5  # 缩短单次采集时间至1.5秒，避免疲劳
        
        try:
            time.sleep(1) # 准备时间
            
            all_data = [] # List[(n_channels, n_points)]
            all_labels = [] # List[label]

            total_steps = trials * len(self.sti_lst)
            current_step = 0

            for trial in range(trials):
                #打乱顺序防止适应
                indices = list(range(len(self.sti_lst)))
                np.random.shuffle(indices)
                
                for idx in indices:
                    current_step += 1
                    command_name = self.commands[idx]
                    freq = self.sti_lst[idx]
                    
                    # 提示阶段 (Cue) - 缩短至1.5秒
                    self.update_text.emit(f"[训练 {current_step}/{total_steps}] 请注视: {command_name} ({freq}Hz)")
                    self.setCue(idx)
                    try: pyttsx3.speak(f"看{command_name}") 
                    except: pass
                    time.sleep(1.5)
                    
                    # 刺激采集阶段 - calibration_time 秒
                    self.start_cache = True
                    self.cache_data = np.array([])
                    start_time = time.time()
                    
                    # 清空进度条
                    self.update_progress.emit(0)
                    
                    # 采集循环
                    while not self.finish:
                        now = time.time()
                        target_rect = self.sti_rects[idx]
                        
                        # 闪烁
                        for r_idx, r in enumerate(self.sti_rects):
                            r.changeColor(r.sti, now - start_time)

                        current_len = 0
                        if len(self.cache_data) > 0:
                            current_len = self.cache_data.shape[-1]
                            
                        target_len = calibration_time * 250
                        prog_val = (current_len / target_len) * 100
                        self.update_progress.emit(int(prog_val))
                        
                        if current_len >= target_len:
                            break
                            
                        time.sleep(0.005)

                    if self.finish: break
                    
                    # 停止采集
                    self.start_cache = False
                    self.setDefaultColor()
                    
                    # 保存数据
                    if len(self.cache_data) > 0:
                        # 截取最后 calibration_time 秒的数据
                         used_len = int(calibration_time * 250)
                         # 确保维度正确 (n_channels, n_points)
                         if self.cache_data.shape[-1] >= used_len:
                             snippet = self.cache_data[:, -used_len:]
                             all_data.append(snippet)
                             all_labels.append(idx)
                    
                    # 休息阶段 - 缩短至0.5秒
                    self.update_text.emit("休息...")
                    time.sleep(0.5)

            if self.finish:
                self.update_text.emit("训练已取消")
                self.train_btn.setEnabled(True)
                return

            # 开始训练模型
            self.update_text.emit("正在计算模型参数...")
            if len(all_data) > 0:
                # 堆叠数据 (trials, channels, points)
                try:
                    # 确保所有数据长度一致
                    min_len = min([d.shape[-1] for d in all_data])
                    X_list = [d[:, :min_len] for d in all_data]
                    X = np.stack(X_list)
                    y = np.array(all_labels)
                    
                    # 1. 保存原始校准数据到磁盘，方便后续分析
                    save_dir = os.path.join('saveCarData', config.subjectName, 'calibration')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    timestamp = time.strftime('%H%M%S')
                    save_path = os.path.join(save_dir, f'calib_{timestamp}.mat')
                    savemat(save_path, {'X': X, 'y': y, 'fs': 250})
                    print(f"校准数据已保存: {save_path}")

                    # 2. 累积历史数据 (解决'越训练越差'问题)
                    # 将本次数据加入历史缓冲区
                    
                    # 简化逻辑：每次都直接 append，在合并时再处理长度
                    if not hasattr(self, 'history_X_list'):
                        self.history_X_list = []
                        self.history_y_list = []
                        
                    self.history_X_list.append(X) # 这里的 X 是本轮的 (n_trials, n_channels, points)
                    self.history_y_list.append(y)
                    
                    # 寻找全局最小长度
                    global_min_len = min([x.shape[-1] for x in self.history_X_list])
                    
                    # 统一截断并合并
                    final_X_list = [x[..., :global_min_len] for x in self.history_X_list]
                    train_X = np.concatenate(final_X_list, axis=0) # (Total_trials, channels, points)
                    train_y = np.concatenate(self.history_y_list, axis=0)
                    
                    # 调用模型训练
                    self.fbcca.fit(train_X, train_y)
                    
                    msg = f"模型训练完成！累计样本数: {train_X.shape[0]}"
                    self.update_text.emit(msg)
                    print(f"训练完成: Total X shape {train_X.shape}")
                    try: pyttsx3.speak("训练完成")
                    except: pass
                except Exception as e:
                    self.update_text.emit(f"训练失败: {str(e)}")
                    print(f"Training Error: {e}")
            else:
                self.update_text.emit("未采集到有效数据")

        except Exception as e:
            print(f"Calibration Error: {e}")
            self.update_text.emit("训练出错")
        finally:
            self.train_btn.setEnabled(True)
            self.setDefaultColor()
            self.finish = True
            self.start_flick = False

    def start_sti_event(self):
        if not self.start_flick:
             # 如果模型没训练，提示一下
            if not self.fbcca.is_fitted:
                 reply = QMessageBox.question(self, '模型未训练', 
                                            "当前模型使用的是默认参数或未训练，识别准确率可能较低。\n建议先点击【开始模型训练】。\n是否仍要继续？",
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                 if reply == QMessageBox.No:
                     return

            # 开始连续刺激
            self.start_flick = True
            self.finish = False
            self.continuous_mode = True
            self.start_cache = False
            self.cache_data = np.array([])
            self.show_label.setText('连续刺激模式 - 再按回车键停止')
            th = threading.Thread(target=self.flick)
            th.start()
        else:
            # 停止连续刺激
            self.finish = True
            self.continuous_mode = False
            self.start_cache = False
            self.setDefaultColor()
            self.show_label.setText('已停止')

    def flick(self):
        # 连续刺激执行
        while not self.finish:
            # 单次刺激
            start_time = time.time()
            self.cache_data = np.array([])
            self.start_cache = True
            
            # 显示“识别中”状态
            self.update_text.emit("识别中...")
            
            # 刺激 time * 250 (原0.5s，增加到1.5s以提高准确率)
            detect_time = 1.0 # 再次调整为1.0s，兼顾速度和准确率
            current_times = detect_time 
            
            while self.cache_data.shape[-1] < current_times * 250 and not self.finish:
                end_time = time.time()
                for idx, rect in enumerate(self.sti_rects):
                    rect.changeColor(rect.sti, end_time - start_time)

                self.update_progress.emit(int((self.cache_data.shape[-1] / (current_times * 250)) * 100))
                time.sleep(0.00001)

            self.update_progress.emit(100)
            self.setDefaultColor()
            self.start_cache = False
            
            if len(self.cache_data) != 0:
                # 截取合适长度用于分类
                used_len = int(current_times * 250)
                if self.cache_data.shape[-1] >= used_len:
                    used_data = self.cache_data[:, -used_len:]
                    
                    # 添加数据保存功能
                    savePath = os.path.join('saveCarData', config.subjectName)
                    if not os.path.exists(savePath):
                        os.makedirs(savePath)

                    fileNums = len(glob(os.path.join(savePath, '*.mat')))
                    saveFile = os.path.join(savePath, f'{fileNums + 1}.mat')
                    savemat(saveFile, {'data': used_data})
                    
                    self.cache_data = np.array([])
                    
                    # 分类
                    try:
                        result = self.fbcca.classify(used_data)
                        result = int(result)
                        self.set_result(result)
                    except Exception as e:
                        print(f"Classify Error: {e}")
            
            # 等待1秒后开始下一次刺激（如果还在连续模式中）
            if not self.finish:
                wait_start = time.time()
                while time.time() - wait_start < 0.5 and not self.finish:
                    time_left = 0.5 - (time.time() - wait_start)
                    self.update_text.emit(f"冷却中 {time_left:.1f}s")
                    time.sleep(0.01)
        
        # 退出循环后停止
        self.start_flick = False

    def set_result(self, idx):
        """
        处理识别结果
        
        参数:
            idx: 识别结果索引（0-4）
                0: 前进 (8Hz)
                1: 后退 (9Hz)
                2: 左转 (9.5Hz)
                3: 停止 (10Hz)
                4: 右转 (10.5Hz)
        """
        if idx < len(self.commands):
            # 使用Signal更新UI
            command = self.commands[idx]
            try: pyttsx3.speak(command)
            except: pass
            
            self.update_text.emit(f"执行命令: {command}")
            
            # 打印识别结果和对应的刺激频率
            print(f"=" * 50)
            print(f"识别结果: {command}")
            print(f"命令索引: {idx}")
            print(f"刺激频率: {self.sti_lst[idx]} Hz")
            try:
                with open(self.exp_txt_path, "a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%H:%M:%S')}] 判定结果: {command}, 频率: {self.sti_lst[idx]}Hz, 指令索引: {idx}\n")
            except Exception: pass
            print(f"=" * 50)
            
            # 发送命令到小车（使用持久化连接）
            try:
                # 检查连接状态并自动重连（由RobotClient内部处理）
                self.robot_client.move(idx)
                print(f"✓ Socket命令处理完成: {command}")
            except Exception as e:
                print(f"✗ 发送命令失败: {str(e)}")
            
        else:
            print(f"识别结果索引超出范围: {idx}")

    def getData(self, data):
        if self.finish:
            return

        if self.start_cache:
            if len(self.cache_data) == 0:
                self.cache_data = data
            else:
                self.cache_data = np.concatenate([self.cache_data, data], axis=-1)

    def keyPressEvent(self, event):
        # 回车键：开始/停止连续刺激
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.start_sti_event()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if hasattr(self, 'video_panel') and self.video_panel is not None:
            self.video_panel.close()
        
        # 关闭小车连接
        if hasattr(self, 'robot_client'):
            try:
                self.robot_client.close()
            except: pass
            
        try:
            super().closeEvent(event)
        except Exception:
            pass

def test_car_control():
    """
    测试脑控小车Socket控制
    在终端输入0-4的指令，通过Socket发送到小车
    """
    print("\n" + "=" * 60)
    print("脑控小车Socket测试系统".center(60))
    print("=" * 60)
    
    # 输入IP地址
    ip = input(f"\n请输入小车IP地址 (默认: {ROBOT_IP}): ").strip()
    if ip == '':
        ip = ROBOT_IP
    
    # 输入端口
    port_input = input(f"请输入端口号 (默认: {PORT}): ").strip()
    if port_input == '':
        port = PORT
    else:
        try:
            port = int(port_input)
        except ValueError:
            print(f"端口号无效，使用默认端口: {PORT}")
            port = PORT
    
    # 连接小车
    print(f"\n正在连接小车 {ip}:{port}...")
    client = RobotClient(ip, port)
    client.connect()
    
    if not client.connected:
        print("✗ 连接失败，请检查:")
        print("  1. 小车是否开机")
        print("  2. 小车和电脑是否在同一网络")
        print("  3. IP地址和端口是否正确")
        return
    
    print("✓ 小车连接成功!")
    
    # 命令映射
    commands = {
        0: "前进",
        1: "后退",
        2: "左转",
        3: "停止",
        4: "右转",
    }
    
    # 显示命令说明
    print("\n" + "=" * 60)
    print("命令列表:")
    print("-" * 60)
    for idx, name in commands.items():
        print(f"  [{idx}] {name}")
    print("-" * 60)
    print("输入指令编号 (0-4) 或 'q' 退出")
    print("=" * 60 + "\n")
    
    # 命令输入循环
    try:
        while True:
            user_input = input("请输入指令 [0-4] 或 q 退出: ").strip()
            
            # 退出
            if user_input.lower() == 'q':
                print("\n正在退出...")
                break
            
            # 检查输入是否有效
            try:
                command_idx = int(user_input)
                
                if command_idx not in commands:
                    print(f"✗ 错误: 指令必须在 0-4 之间，您输入的是 {command_idx}")
                    continue
                
                # 获取命令名称
                command_name = commands[command_idx]
                
                # 发送命令到小车
                print(f"\n>>> 发送命令: [{command_idx}] {command_name}")
                
                try:
                    client.move(command_idx)
                    print(f"✓ 命令已发送")
                except Exception as e:
                    print(f"✗ 发送失败: {str(e)}")
                
                print()  # 空行分隔
                
            except ValueError:
                print(f"✗ 错误: 请输入有效的数字 (0-4) 或 'q'")
                
    except KeyboardInterrupt:
        print("\n\n检测到 Ctrl+C，正在退出...")
    
    finally:
        # 关闭连接
        print("\n正在关闭连接...")
        client.close()
        print("✓ 连接已关闭")
        
        print("\n" + "=" * 60)
        print("测试结束".center(60))
        print("=" * 60 + "\n")

if __name__ == "__main__":
    test_car_control()

