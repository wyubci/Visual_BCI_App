import socket
import pyttsx3
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from qfluentwidgets import *
import threading
import numpy as np
import time
from scipy import signal
from scipy.io import savemat
from glob import glob
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.FBCCA import FBCCA
from config import config

# ==========================================
# PC CLIENT FOR X1
# Run this on Windows
# ==========================================

ROBOT_IP = '192.168.1.11'  # CHANGE THIS IF NEEDED
PORT = 65432

class RobotClient:
    def __init__(self, ip, port):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False

    def connect(self):
        try:
            print(f"Connecting to {self.addr}...")
            self.sock.connect(self.addr)
            self.connected = True
            print("Connected!")
        except Exception as e:
            print(f"Connection Failed: {e}")

    def send(self, cmd):
        if not self.connected: return
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
            v1=61
            v2=0
            v3=64
            v4=5
            self.set_motor(v1, v2, v3, v4)
            time.sleep(2.68)
            self.set_motor(0,0,0,0)
        elif(ing==1):
            v1=-58
            v2=0
            v3=-80
            v4=0
            self.set_motor(v1, v2, v3, v4)
            time.sleep(2.8)
            self.set_motor(0,0,0,0)
        elif(ing==2):
            v1=-65
            v2=0
            v3=65
            v4=0
            self.set_motor(v1, v2, v3, v4)
            time.sleep(3.165)
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
            time.sleep(3.42)
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
        painter.setPen(self.current_color)
        painter.setBrush(self.current_color)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)

        painter.setBrush(QBrush())
        pen = QPen()
        pen.setWidth(2)
        pen.setColor(QColor(255, 0, 0))
        painter.setPen(pen)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)

        text_color = QColor(0, 0, 0)
        font = QFont()
        font.setPixelSize(self.fontSize)

        pen = QPen()
        pen.setColor(text_color)
        pen.setBrush(text_color)

        painter.setPen(pen)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(1, 1, -1, -1), Qt.AlignCenter, self.text)

        painter.end()

class CarControlWindow(QWidget):
    def __init__(self, objectName):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setObjectName(objectName)
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
        
        self._initLayout()
        self._initItems()
        
        self.times = 4
        self.fbcca = FBCCA(3, self.times, self.sti_lst)
        
        # 设置焦点策略，使窗口可以接收键盘事件
        self.setFocusPolicy(Qt.StrongFocus)

    def setQss(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        qss_path = os.path.join(base_dir, 'source', 'qss', 'mainWindow.qss')
        with open(qss_path, encoding='utf-8') as f:
            self.setStyleSheet(f.read())
        
        # 添加最黑的背景色设置
        self.setStyleSheet(self.styleSheet() + "QWidget#carControlWindow { background-color: #000000; }")

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
        self.layout.addSpacing(40)
        
        # 刺激区域布局
        self.sti_rects = []
        
        # 创建垂直布局容器
        self.mainVLayout = QVBoxLayout()
        self.mainVLayout.setContentsMargins(50, 50, 50, 50)
        self.mainVLayout.setSpacing(180)  # 增加第一行和第二行之间的垂直距离
        self.layout.addLayout(self.mainVLayout)
        
        # 第一行布局 - 前进、后退
        self.row1Layout = QHBoxLayout()
        self.row1Layout.setContentsMargins(0, 0, 0, 0)
        self.row1Layout.setSpacing(280)  # 增加前进和后退之间的水平距离
        self.mainVLayout.addLayout(self.row1Layout)
        
        # 第二行布局 - 左转、停止、右转
        self.row2Layout = QHBoxLayout()
        self.row2Layout.setContentsMargins(0, 0, 0, 0)
        self.row2Layout.setSpacing(150)
        self.mainVLayout.addLayout(self.row2Layout)
        
        rect_size = 200  # 刺激区域大小
        
        # 第一行 - 前进、后退
        # 添加左侧空白，使第一行居中
        self.row1Layout.addStretch()
        
        # 前进 (8Hz)
        self.forward_rect = StiRect("前进", self, self.sti_lst[0])
        self.forward_rect.setFixedSize(rect_size, rect_size)
        self.row1Layout.addWidget(self.forward_rect)
        self.sti_rects.append(self.forward_rect)
        
        # 后退 (9Hz)
        self.backward_rect = StiRect("后退", self, self.sti_lst[1])
        self.backward_rect.setFixedSize(rect_size, rect_size)
        self.row1Layout.addWidget(self.backward_rect)
        self.sti_rects.append(self.backward_rect)
        
        # 添加右侧空白，使第一行居中
        self.row1Layout.addStretch()
        
        # 第二行 - 左转、停止、右转
        # 左转 (9.5Hz)
        self.left_rect = StiRect("左转", self, self.sti_lst[2])
        self.left_rect.setFixedSize(rect_size, rect_size)
        self.row2Layout.addWidget(self.left_rect)
        self.sti_rects.append(self.left_rect)
        
        # 停止 (10Hz)
        self.stop_rect = StiRect("停止", self, self.sti_lst[3])
        self.stop_rect.setFixedSize(rect_size, rect_size)
        self.row2Layout.addWidget(self.stop_rect)
        self.sti_rects.append(self.stop_rect)
        
        # 右转 (10.5Hz)
        self.right_rect = StiRect("右转", self, self.sti_lst[4])
        self.right_rect.setFixedSize(rect_size, rect_size)
        self.row2Layout.addWidget(self.right_rect)
        self.sti_rects.append(self.right_rect)
        
        self.layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Maximum, QSizePolicy.Expanding))
    
    def setDefaultColor(self):
        for rect in self.sti_rects:
            rect.setDefaultColor()

    def start_sti_event(self):
        if not self.start_flick:
            self.start_flick = True
            self.finish = False
            self.start_cache = False
            self.cache_data = np.array([])
            self.show_label.setText('')
            th = threading.Thread(target=self.flick)
            th.start()
        else:
            # 如果正在刺激中，停止刺激
            self.start_flick = False
            self.finish = True
            self.start_cache = False
            self.setDefaultColor()
            self.show_label.setText('')

    def flick(self):
        # 单次刺激执行
        if self.finish:  # 如果被手动停止，直接返回
            return

        start_time = time.time()
        self.cache_data = np.array([])
        self.start_cache = True
        # 刺激4秒
        while self.cache_data.shape[-1] < self.times * 250 and not self.finish:
            end_time = time.time()
            for idx, rect in enumerate(self.sti_rects):
                rect.changeColor(rect.sti, end_time - start_time)

            self.progress.setVal((self.cache_data.shape[-1] / (self.times * 250)) * 100)
            time.sleep(0.00001)

        self.progress.setVal(100)
        self.setDefaultColor()
        self.start_cache = False
        
        # 刺激完成后自动停止
        self.start_flick = False
        self.finish = True
        
        if len(self.cache_data) != 0:
            used_data = self.cache_data[:, -1000:]
            
            # 添加数据保存功能
            savePath = os.path.join('saveCarData', config.subjectName)
            if not os.path.exists(savePath):
                os.makedirs(savePath)

            fileNums = len(glob(os.path.join(savePath, '*.mat')))
            saveFile = os.path.join(savePath, f'{fileNums + 1}.mat')
            savemat(saveFile, {'data': used_data})
            
            self.cache_data = np.array([])
            result = self.fbcca.classify(used_data)
            result = int(result)
            self.set_result(result)

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
            command = self.commands[idx]
            pyttsx3.speak(command)
            self.show_label.setText(f"执行命令: {command}")
            
            # 打印识别结果和对应的刺激频率
            print(f"=" * 50)
            print(f"识别结果: {command}")
            print(f"命令索引: {idx}")
            print(f"刺激频率: {self.sti_lst[idx]} Hz")
            print(f"=" * 50)
            
            # 发送命令到小车（通过Socket）
            try:
                client = RobotClient(ROBOT_IP, PORT)
                client.connect()
                if client.connected:
                    client.move(idx)
                    print(f"✓ Socket命令已发送: {command}")
                else:
                    print(f"✗ 小车连接失败")
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
        # 空格键或回车键：开始刺激
        if event.key() == Qt.Key_Space or event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if not self.start_flick:  # 只有在未刺激时才开始
                self.show_label.setText("开始刺激识别")
                self.start_sti_event()
        super().keyPressEvent(event)

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

