import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PyQt5.QtCore import QObject, pyqtSignal
import scipy.signal as signal

class TSception(nn.Module):
    """
    TSception模型 - 专为SSVEP信号设计的深度学习模型
    结合了时间卷积和空间卷积，能有效捕捉SSVEP信号中的频率特征
    """
    def __init__(self, num_classes=8, num_channels=8, input_size=1000, sampling_rate=250, num_T=15, num_S=15, hid_channels=32, dropout_rate=0.5):
        super(TSception, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.input_size = input_size
        self.sampling_rate = sampling_rate
        
        # 定义模型参数
        self.num_T = num_T
        self.num_S = num_S
        self.hid_channels = hid_channels
        self.dropout_rate = dropout_rate
        
        # 定义时间卷积的尺寸参数
        # 创建不同尺寸的卷积核，用于捕捉不同频率的时间模式
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        
        # 时间卷积层 - 捕捉不同时间尺度上的特征
        self.Tception1 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[0] * sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)))
        
        self.Tception2 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[1] * sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)))
        
        self.Tception3 = nn.Sequential(
            nn.Conv2d(1, num_T, kernel_size=(1, int(self.inception_window[2] * sampling_rate)), stride=1, padding=0),
            nn.BatchNorm2d(num_T),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)))
        
        # 空间卷积层 - 捕捉通道间的空间关系
        self.Sception = nn.Sequential(
            nn.Conv2d(num_T*3, num_S, kernel_size=(num_channels, 1), stride=1, padding=0),
            nn.BatchNorm2d(num_S),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)))
        
        # 计算全连接层的输入尺寸
        self.len_after_concat = self._get_final_flattened_size()
        
        # 全连接层和分类层
        self.fc = nn.Sequential(
            nn.Linear(self.len_after_concat, hid_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_channels, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        # 输入形状调整: (batch_size, channels, data_length) -> (batch_size, 1, channels, data_length)
        x = x.unsqueeze(1)
        
        # 时间卷积
        y1 = self.Tception1(x)
        y2 = self.Tception2(x)
        y3 = self.Tception3(x)
        
        # 合并时间卷积的输出
        out = torch.cat((y1, y2, y3), dim=1)
        
        # 空间卷积
        out = self.Sception(out)
        
        # 展平
        out = out.view(out.size(0), -1)
        
        # 全连接层进行分类
        out = self.fc(out)
        
        return out
    
    def _get_final_flattened_size(self):
        """计算展平后的特征维度，用于全连接层的设计"""
        # 创建一个假的输入数据
        with torch.no_grad():
            x = torch.zeros(1, 1, self.num_channels, self.input_size)
            
            # 前向计算每一层的输出大小
            y1 = self.Tception1(x)
            y2 = self.Tception2(x)
            y3 = self.Tception3(x)
            out = torch.cat((y1, y2, y3), dim=1)
            out = self.Sception(out)
            
            # 返回展平后的维度
            return out.view(out.size(0), -1).size(1)

class DeepLearning(QObject):
    """
    深度学习SSVEP分类器，提供与FBCCA类似的接口
    使用TSception模型进行SSVEP信号分类
    """
    # 定义信号，用于发送识别结果
    sendResultSignal = pyqtSignal(object)
    
    def __init__(self, targets=[8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5], Nh=8, sampling_rate=250):
        """
        初始化深度学习模型
        
        参数:
            targets: 目标频率列表，默认为[8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]
            Nh: 通道数量，默认为8
            sampling_rate: 采样率，默认为250
        """
        super(DeepLearning, self).__init__()
        self.targets = targets
        self.Nf = len(targets)  # 目标频率数量
        self.Nh = Nh  # 通道数量
        self.Fs = sampling_rate  # 采样率(Hz)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型
        self.model = TSception(
            num_classes=self.Nf,
            num_channels=self.Nh,
            input_size=1000,  # 假设输入为1000个采样点
            sampling_rate=self.Fs,
            num_T=15,
            num_S=15,
            hid_channels=32,
            dropout_rate=0.5
        ).to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        
        # 模型路径
        self.model_path = "models/ssvep_tsception_model.pth"
        
        # 尝试加载预训练模型
        try:
            self.load_model()
            print("预训练模型加载成功")
        except:
            print("未找到预训练模型，需要先训练模型")

    def preprocess(self, eeg_data):
        """
        对EEG数据进行预处理
        
        参数:
            eeg_data: 原始脑电信号，形状为[通道数, 采样点数]
            
        返回:
            processed_data: 预处理后的张量，形状为[1, 通道数, 采样点数]
        """
        # 带通滤波 (5-90Hz)
        nyq = self.Fs / 2
        b, a = signal.butter(4, [5/nyq, 90/nyq], btype='bandpass')
        filtered_data = signal.filtfilt(b, a, eeg_data)
        
        # 标准化
        mean = np.mean(filtered_data, axis=1, keepdims=True)
        std = np.std(filtered_data, axis=1, keepdims=True)
        normalized_data = (filtered_data - mean) / (std + 1e-8)
        
        # 转换为PyTorch张量
        tensor_data = torch.FloatTensor(normalized_data).unsqueeze(0)
        
        return tensor_data

    def train(self, train_data, train_labels, epochs=50, batch_size=32, lr=0.001):
        """
        训练模型
        
        参数:
            train_data: 训练数据，形状为[样本数, 通道数, 采样点数]
            train_labels: 训练标签，形状为[样本数]
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
        """
        # 确保模型处于训练模式
        self.model.train()
        
        # 转换为张量
        X_train = torch.FloatTensor(train_data)
        y_train = torch.LongTensor(train_labels)
        
        # 创建数据集和数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # 开始训练
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                
                # 参数更新
                optimizer.step()
                
                # 统计
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # 更新学习率
            scheduler.step()
            
            # 打印训练信息
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
        
        # 保存模型
        self.save_model()
        
        # 设置为评估模式
        self.model.eval()
        
        print("模型训练完成并保存")
    
    def save_model(self):
        """保存模型"""
        torch.save(self.model.state_dict(), self.model_path)
    
    def load_model(self):
        """加载模型"""
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
    
    def recognize(self, test_data):
        """
        对测试数据进行SSVEP频率识别
        
        参数:
            test_data: 测试脑电数据，形状为[通道数, 采样点数]
            
        返回:
            result: 识别结果的索引，对应目标频率列表中的位置
        """
        # 确保模型处于评估模式
        self.model.eval()
        
        # 数据预处理
        processed_data = self.preprocess(test_data)
        
        # 将数据移到相应设备
        processed_data = processed_data.to(self.device)
        
        # 禁用梯度计算
        with torch.no_grad():
            # 模型预测
            outputs = self.model(processed_data)
            _, predicted = torch.max(outputs, 1)
            result = predicted.item()
        
        # 发送结果信号
        # self.sendResultSignal.emit(result)
        
        return result
