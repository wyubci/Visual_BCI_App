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
from models.TDCA import TDCA
from .drone_control import DroneController
import os
from config import config

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
        # self.subjectName = "TestSubject"

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

class DroneControlWindow(QWidget):
    def __init__(self, objectName):
        super().__init__()
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMouseTracking(True)
        self.setObjectName(objectName)
        self.setQss()
        
        # 设置刺激频率列表 - 修改为8个频率
        self.sti_lst = [
            8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5
        ]
        self.commands = [
            "起飞", "降落", "上升", "下降", "前进", "后退", "左移", "右移"
        ]
        
        # 初始化无人机控制器
        self.drone_controller = DroneController()
        self.drone_controller.initialize()
        
        self.cache_data = np.array([])
        self.start_flick = False
        self.finish = True
        self.start_cache = False
        
        # 添加刺激模式标志
        self.is_m_mode = False  # True为M键模式(4分类), False为空格键模式(12分类)
        
        self._initLayout()
        self._initItems()
        
        self.times = 1
        from config import config
        nh = len(getattr(config, 'lsl_selected_channels', list(range(22, 31)))) if getattr(config, 'device_type', 'neuro_dance_tcp') == 'lsl' else 8
        self.fbcca = TDCA(3, self.times, self.sti_lst, Nh=nh)
        
        # 设置焦点策略，使窗口可以接收键盘事件
        self.setFocusPolicy(Qt.StrongFocus)

    def setQss(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        qss_path = os.path.join(base_dir, 'source', 'qss', 'mainWindow.qss')
        with open(qss_path, encoding='utf-8') as f:
            self.setStyleSheet(f.read())
        
        # 添加最黑的背景色设置
        self.setStyleSheet(self.styleSheet() + "QWidget#droneControlWindow { background-color: #000000; }")

    def _initLayout(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 10, 20, 20)  # 减小上边距
        self.layout.setSpacing(1)
        self.setLayout(self.layout)

    def _initItems(self):
        self.show_label = DisplayLabel()
        self.show_label.setText("脑控无人机界面")
        self.show_label.setFixedHeight(80)  # 增加高度以容纳多行文本
        # 设置标题字体较小
        font = QFont()
        font.setPixelSize(24)  # 调整字体大小
        self.show_label.setFont(font)
        self.layout.addWidget(self.show_label)

        self.progress = ProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)
        self.layout.addSpacing(20)  # 减小间距
        
        # 刺激区域布局
        self.sti_rects = []
        
        # 创建水平布局，左右两侧
        self.mainHLayout = QHBoxLayout()
        self.mainHLayout.setContentsMargins(5, 5, 5, 5)
        self.mainHLayout.setSpacing(120)  
        self.layout.addLayout(self.mainHLayout)
        
        # 左侧布局 - 基本控制
        self.leftLayout = QGridLayout()
        self.leftLayout.setContentsMargins(20, 30, 20, 20)  # 增加上边距，让控制区域向下移动
        self.leftLayout.setVerticalSpacing(290)  # 增加垂直间距，让下面一行向下移动
        self.leftLayout.setHorizontalSpacing(260)
        
        # 右侧布局 - 方向控制
        self.rightLayout = QGridLayout()
        self.rightLayout.setContentsMargins(20, 30, 20, 20)  # 增加上边距，让控制区域向下移动
        self.rightLayout.setVerticalSpacing(290)  # 增加垂直间距，让下面一行向下移动
        self.rightLayout.setHorizontalSpacing(260)
        
        # 添加到主布局
        self.mainHLayout.addLayout(self.leftLayout)
        self.mainHLayout.addLayout(self.rightLayout)
        
        rect_size = 160  # 减小刺激区域大小
        
        # 左侧布局 - 无人机基本控制
        # 起飞 - 左上
        self.takeoff_rect = StiRect("起飞", self, self.sti_lst[0])
        self.takeoff_rect.setFixedSize(rect_size, rect_size)
        self.leftLayout.addWidget(self.takeoff_rect, 0, 0)
        self.sti_rects.append(self.takeoff_rect)
        
        # 降落 - 右上
        self.land_rect = StiRect("降落", self, self.sti_lst[1])
        self.land_rect.setFixedSize(rect_size, rect_size)
        self.leftLayout.addWidget(self.land_rect, 0, 1)
        self.sti_rects.append(self.land_rect)
        
        # 上升 - 左下
        self.up_rect = StiRect("上升", self, self.sti_lst[2])
        self.up_rect.setFixedSize(rect_size, rect_size)
        self.leftLayout.addWidget(self.up_rect, 1, 0)
        self.sti_rects.append(self.up_rect)
        
        # 下降 - 右下
        self.down_rect = StiRect("下降", self, self.sti_lst[3])
        self.down_rect.setFixedSize(rect_size, rect_size)
        self.leftLayout.addWidget(self.down_rect, 1, 1)
        self.sti_rects.append(self.down_rect)
        
        # 右侧布局 - 方向控制
        # 前进 - 左上
        self.forward_rect = StiRect("前进", self, self.sti_lst[4])
        self.forward_rect.setFixedSize(rect_size, rect_size)
        self.rightLayout.addWidget(self.forward_rect, 0, 0)
        self.sti_rects.append(self.forward_rect)
        
        # 后退 - 右上
        self.backward_rect = StiRect("后退", self, self.sti_lst[5])
        self.backward_rect.setFixedSize(rect_size, rect_size)
        self.rightLayout.addWidget(self.backward_rect, 0, 1)
        self.sti_rects.append(self.backward_rect)
        
        # 左移 - 左下
        self.left_rect = StiRect("左移", self, self.sti_lst[6])
        self.left_rect.setFixedSize(rect_size, rect_size)
        self.rightLayout.addWidget(self.left_rect, 1, 0)
        self.sti_rects.append(self.left_rect)
        
        # 右移 - 右下
        self.right_rect = StiRect("右移", self, self.sti_lst[7])
        self.right_rect.setFixedSize(rect_size, rect_size)
        self.rightLayout.addWidget(self.right_rect, 1, 1)
        self.sti_rects.append(self.right_rect)
        
        # 删除第三行刺激区域 - 现在只保留8个刺激（前两行）
        
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
        while not self.finish:
            start_time = time.time()
            self.cache_data = np.array([])
            self.start_cache = True
            while self.cache_data.shape[-1] < self.times * 250 and not self.finish:
                end_time = time.time()
                for idx, rect in enumerate(self.sti_rects):
                    rect.changeColor(rect.sti, end_time - start_time)

                self.progress.setValue((self.cache_data.shape[-1] / (self.times * 250)) * 100)
                time.sleep(0.00001)

            self.progress.setValue(100)
            self.setDefaultColor()
            self.start_cache = False

            if not self.finish and len(self.cache_data) != 0:
                used_data = self.cache_data[:, -self.times * 250:]

                savePath = os.path.join('saveDroneData', config.subjectName)
                if not os.path.exists(savePath):
                    os.makedirs(savePath)

                fileNums = len(glob(os.path.join(savePath, '*.mat')))
                saveFile = os.path.join(savePath, f'{fileNums + 1}.mat')
                savemat(saveFile, {'data': used_data})

                self.cache_data = np.array([])
                if self.is_m_mode:
                    result = self.fbcca.classify_4_class(used_data)
                else:
                    result = self.fbcca.classify(used_data)
                result = int(result)
                self.set_result(result)
            time.sleep(0.01)

        self.start_flick = False

    def set_result(self, idx):
        if self.is_m_mode:
            # M键模式：只识别4分类（前进、后退、左移、右移）
            # 映射到对应的动作索引：4-前进, 5-后退, 6-左移, 7-右移
            m_mode_mapping = {
                0: 4,  # 第1个频率 -> 前进
                1: 5,  # 第2个频率 -> 后退  
                2: 6,  # 第3个频率 -> 左移
                3: 7,  # 第4个频率 -> 右移
            }
            
            # 只处理前4个识别结果
            if idx < 4:
                action_idx = m_mode_mapping[idx]
                command = self.commands[action_idx]
                pyttsx3.speak(command)
                self.show_label.setText(f"执行命令(M模式): {command}")
                
                # 使用M模式的距离设置
                distance_map = {
                    4: config.m_mode_distances['forward'],   # 前进
                    5: config.m_mode_distances['backward'],  # 后退
                    6: config.m_mode_distances['left'],      # 左移
                    7: config.m_mode_distances['right']      # 右移
                }
                
                try:
                    self.drone_controller.execute_action(action_idx, distance=distance_map[action_idx])
                    print(f"发送无人机命令(M模式): {command}, 距离: {distance_map[action_idx]}cm")
                except Exception as e:
                    print(f"发送命令失败: {str(e)}")
            else:
                print(f"M模式下忽略识别结果: {idx}")
        else:
            # 空格键模式：识别全部8分类
            command = self.commands[idx]
            pyttsx3.speak(command)
            self.show_label.setText(f"执行命令: {command}")
            
            # 发送命令到无人机
            try:
                self.drone_controller.execute_action(idx)
                print(f"发送无人机命令: {command}")
            except Exception as e:
                print(f"发送命令失败: {str(e)}")

    def getData(self, data):
        if self.finish:
            return

        if self.start_cache:
            if len(self.cache_data) == 0:
                self.cache_data = data
            else:
                self.cache_data = np.concatenate([self.cache_data, data], axis=-1)

    def keyPressEvent(self, event):
        # 空格键：8分类模式
        if event.key() == Qt.Key_Space:
            if not self.start_flick:  # 只有在未刺激时才开始
                self.is_m_mode = False
                self.show_label.setText("空格键模式 - 8分类识别")
                self.start_sti_event()
        # M键：4分类模式
        elif event.key() == Qt.Key_M:
            if not self.start_flick:  # 只有在未刺激时才开始
                self.is_m_mode = True
                self.show_label.setText("M键模式 - 4分类识别(前进/后退/左移/右移)")
                self.start_sti_event()
        # 回车键：兼容旧版本，使用8分类模式
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if not self.start_flick:  # 只有在未刺激时才开始
                self.is_m_mode = False
                self.show_label.setText("回车键模式 - 8分类识别")
                self.start_sti_event()
        super().keyPressEvent(event)

    def set_m_mode_distances(self, forward=None, backward=None, left=None, right=None):
        """
        设置M键模式下的移动距离
        
        参数:
            forward: 前进距离(厘米)
            backward: 后退距离(厘米)  
            left: 左移距离(厘米)
            right: 右移距离(厘米)
        """
        if forward is not None:
            config.m_mode_distances['forward'] = forward
        if backward is not None:
            config.m_mode_distances['backward'] = backward
        if left is not None:
            config.m_mode_distances['left'] = left
        if right is not None:
            config.m_mode_distances['right'] = right
        
        config.save()
        print(f"M键模式距离已更新: {config.m_mode_distances}")

    def get_m_mode_distances(self):
        """
        获取当前M键模式的移动距离设置
        
        返回:
            dict: 包含前进、后退、左移、右移距离的字典
        """
        return config.m_mode_distances.copy()
    
    def set_down_weight(self, weight=0.75):
        """
        设置下降命令的识别权重
        
        参数:
            weight: 权重值，0.1-1.0之间，越小越不容易识别为下降
        """
        self.fbcca.set_frequency_weight(3, weight)  # 索引3对应下降命令
        print(f"下降命令权重已设置为: {weight}")
    
    def get_frequency_weights(self):
        """
        获取当前所有频率的权重设置
        
        返回:
            dict: 频率权重字典，键为命令名称，值为权重
        """
        weights = self.fbcca.get_frequency_weights()
        weight_dict = {}
        for i, command in enumerate(self.commands):
            if i < len(weights):
                weight_dict[command] = weights[i]
        return weight_dict

def test_drone_control_modes():
    """
    测试无人机控制的不同模式
    """
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # 创建无人机控制窗口
    drone_window = DroneControlWindow("droneControlWindow")
    
    # 测试M键模式距离设置
    print("当前M键模式距离:", drone_window.get_m_mode_distances())
    
    # 修改M键模式距离
    drone_window.set_m_mode_distances(forward=80, backward=80, left=80, right=80)
    print("修改后M键模式距离:", drone_window.get_m_mode_distances())
    
    # 显示当前频率权重
    print("当前频率权重:", drone_window.get_frequency_weights())
    
    # 调整下降命令权重（已默认设置为0.75）
    print("下降命令权重已在初始化时设置为0.75，减小误识别概率")
    
    # 可以进一步调整下降权重
    # drone_window.set_down_weight(0.6)  # 进一步减小权重
    
    drone_window.show()
    
    print("测试说明:")
    print("- 按空格键: 执行8分类识别")
    print("- 按M键: 执行4分类识别(前进/后退/左移/右移)")
    print("- 每次按键只执行一次刺激(4秒)")
    print("- 下降命令权重已减小到75%，减少误识别")
    print("- 可通过set_down_weight()方法进一步调整")
    
    app.exec_()

if __name__ == "__main__":
    test_drone_control_modes()