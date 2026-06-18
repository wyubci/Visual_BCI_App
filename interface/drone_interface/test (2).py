import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

# 然后再导入模块
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
from scipy.io import savemat, loadmat
from glob import glob
from models.FBCCA import FBCCA
from drone_control import DroneController
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
        
        # 设置刺激频率列表
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
        
        self._initLayout()
        self._initItems()
        
        self.times = 2
        self.fbcca = FBCCA(3, self.times, self.sti_lst)
        
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
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(1)
        self.setLayout(self.layout)

    def _initItems(self):
        self.show_label = DisplayLabel()
        self.show_label.setText("无人机控制界面")
        self.show_label.setFixedHeight(65)
        self.layout.addWidget(self.show_label)

        self.progress = ProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.layout.addWidget(self.progress)
        self.layout.addSpacing(40)
        
        # 刺激区域布局
        self.sti_rects = []
        
        # 创建水平布局，左右两侧
        self.mainHLayout = QHBoxLayout()
        self.mainHLayout.setContentsMargins(5, 5, 5, 5)
        self.mainHLayout.setSpacing(300)  
        self.layout.addLayout(self.mainHLayout)
        
        # 左侧布局 - 基本控制
        self.leftLayout = QGridLayout()
        self.leftLayout.setContentsMargins(20, 20, 20, 20)  
        self.leftLayout.setSpacing(180)  
        
        # 右侧布局 - 方向控制
        self.rightLayout = QGridLayout()
        self.rightLayout.setContentsMargins(20, 20, 20, 20) 
        self.rightLayout.setSpacing(180)  
        
        # 添加到主布局
        self.mainHLayout.addLayout(self.leftLayout)
        self.mainHLayout.addLayout(self.rightLayout)
        
        rect_size = 150
        
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
        
        self.layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Maximum, QSizePolicy.Expanding))
        
        # 添加测试按钮
        self.test_button = PushButton()
        self.test_button.setText("测试离线数据")
        self.test_button.clicked.connect(self.test_offline_data)
        self.layout.addWidget(self.test_button)

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
            self.start_flick = False
            self.finish = True
            self.start_cache = False
            self.setDefaultColor()
            self.show_label.setText('')

    def flick(self):
        while not self.finish:
            start_time = time.time()
            now_time = 0
            while now_time < 2:
                now_time = time.time() - start_time
                self.progress.setVal(((2 - now_time) / 2) * 100)
                time.sleep(0.00000001)

            start_time = time.time()
            self.cache_data = np.array([])
            self.start_cache = True
            while self.cache_data.shape[-1] < self.times * 250 and not self.finish:
                end_time = time.time()
                for idx, rect in enumerate(self.sti_rects):
                    rect.changeColor(rect.sti, end_time - start_time)

                self.progress.setVal((self.cache_data.shape[-1] / (self.times * 250)) * 100)
                time.sleep(0.00001)

            self.progress.setVal(100)
            self.setDefaultColor()
            self.start_cache = False
            if not self.finish and len(self.cache_data) != 0:
                used_data = self.cache_data[:, -1000:]
                
                # 添加数据保存功能
                savePath = os.path.join('saveDroneData', config.subjectName)
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
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.start_sti_event()
        elif event.key() == Qt.Key_T:  # 按T键测试离线数据
            self.test_offline_data()
        elif event.key() == Qt.Key_E:  # 按E键评估所有数据
            self.evaluate_all_data()
        super().keyPressEvent(event)

    # 添加新的测试离线数据方法
    def test_offline_data(self):
        # 加载saveDroneData/zyz_2目录下的所有.mat文件
        data_dir = os.path.join('saveDroneData', 'xq_5_4')
        mat_files = glob(os.path.join(data_dir, '*.mat'))
        
        # 使用自然排序，确保1.mat排在10.mat之前
        mat_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        print(f"找到{len(mat_files)}个离线数据文件")
        
        for mat_file in mat_files:
            print(f"正在处理文件: {os.path.basename(mat_file)}")
            
            # 加载.mat文件中的数据
            mat_data = loadmat(mat_file)
            if 'data' in mat_data:
                used_data = mat_data['data']
                
                # 使用FBCCA算法识别命令
                result = self.fbcca.classify(used_data)
                result = int(result)
                
                # 显示识别结果并控制无人机
                self.set_result(result)
                
                # 等待一段时间
                time.sleep(3)
            else:
                print(f"文件格式错误: {mat_file}")

    def evaluate_all_data(self):
        """评估xq_all文件夹中的所有数据，计算准确率和混淆矩阵"""
        # 加载saveDroneData/xq_all目录下的所有.mat文件
        data_dir = os.path.join('saveDroneData', 'xq_all')
        mat_files = glob(os.path.join(data_dir, '*.mat'))
        
        # 使用自然排序
        mat_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        print(f"找到{len(mat_files)}个数据文件")
        
        # 用于统计各个标签的准确率
        label_counts = {}
        label_correct = {}
        total_count = 0
        total_correct = 0
        
        # 用于构建混淆矩阵
        true_labels = []
        pred_labels = []
        
        for mat_file in mat_files:
            # 加载.mat文件中的数据
            mat_data = loadmat(mat_file)
            if 'data' in mat_data and 'label' in mat_data:
                # 只使用前2秒的数据(500个采样点)
                used_data = mat_data['data'][:, :650]
                true_label = int(mat_data['label'][0][0])  # 获取真实标签
                
                # 使用FBCCA算法进行识别
                pred_label = int(self.fbcca.classify(used_data))
                
                file_name = os.path.basename(mat_file)
                print(f"文件: {file_name}, 真实标签: {true_label}, 预测标签: {pred_label}, {'正确' if pred_label == true_label else '错误'}")
                
                # 收集标签用于混淆矩阵
                true_labels.append(true_label)
                pred_labels.append(pred_label)
                
                # 更新统计信息
                if true_label not in label_counts:
                    label_counts[true_label] = 0
                    label_correct[true_label] = 0
                
                label_counts[true_label] += 1
                total_count += 1
                
                if pred_label == true_label:
                    label_correct[true_label] += 1
                    total_correct += 1
            else:
                print(f"文件格式错误: {mat_file}")
        
        # 打印各个标签的准确率
        print("\n各标签准确率统计:")
        for label in sorted(label_counts.keys()):
            accuracy = label_correct[label] / label_counts[label] * 100 if label_counts[label] > 0 else 0
            print(f"标签 {label} (命令: {self.commands[label]}): {accuracy:.2f}% ({label_correct[label]}/{label_counts[label]})")
        
        # 打印总准确率
        total_accuracy = total_correct / total_count * 100 if total_count > 0 else 0
        print(f"\n总准确率: {total_accuracy:.2f}% ({total_correct}/{total_count})")
        
        # 计算并打印混淆矩阵
        if true_labels and pred_labels:
            # 获取所有可能的标签
            all_labels = sorted(set(true_labels + pred_labels))
            
            # 创建混淆矩阵
            confusion_mat = np.zeros((len(all_labels), len(all_labels)), dtype=int)
            
            # 填充混淆矩阵
            for t, p in zip(true_labels, pred_labels):
                t_idx = all_labels.index(t)
                p_idx = all_labels.index(p)
                confusion_mat[t_idx, p_idx] += 1
            
            # 打印混淆矩阵（美化版本）
            print("\n混淆矩阵:")
            print("真实\\预测")
            
            # 打印表头
            header = ""
            for label in all_labels:
                header += f"\t{label}"
            print(header)
            
            # 打印分隔线
            separator = "-" * (len(header) + 5)
            print(separator)
            
            # 打印矩阵内容
            for i, true_label in enumerate(all_labels):
                row = f"{true_label}"
                for j in range(len(all_labels)):
                    # 对角线元素（正确分类）加粗显示
                    if i == j:
                        row += f"\t[{confusion_mat[i, j]}]"
                    else:
                        row += f"\t{confusion_mat[i, j]}"
                print(row)

# 测试离线数据
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    
    # 创建无人机控制窗口
    window = DroneControlWindow("testWindow")
    
    # 评估所有数据
    window.evaluate_all_data()
    
    sys.exit(app.exec_())