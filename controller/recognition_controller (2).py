from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import os
from glob import glob
from scipy.io import loadmat

from models.SSVEPNet import SSVEPNet
from config import config
from torch.utils.data import Dataset
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm

class ConfuseDataset(Dataset):
    """自定义数据集"""

    def __init__(self, data):
        self.eeg_data = data[0]
        self.eeg_label = data[1]

    def __len__(self):
        return len(self.eeg_label)

    def __getitem__(self, item):
        eeg_data = self.eeg_data[item]
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0)
        eeg_label = self.eeg_label[item]
        eeg_label = torch.tensor(eeg_label, dtype=torch.int32)

        return eeg_data, eeg_label


class Recognition(QObject):
    trainingProcessSignal = pyqtSignal(object)
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sti_lst = config.sti_lst

        self.__initModel__()

        self.epochs = 200



    def __initModel__(self):
        self.model = SSVEPNet(num_channels=8, T=500, num_classes=38).to(self.device)

    def load_state_dict(self, user):
        checkpointFile = os.path.join(config.userInfoPath, user, 'weight', 'best.pth')

        if os.path.exists(checkpointFile):
            checkpoint = torch.load(checkpointFile, map_location='cpu')
            res = self.model.load_state_dict(checkpoint)

        test_data = np.zeros((8, 500))
        self.predict(test_data)

    def normalized(self, data, axis=2, method='maxmin'):
        if method == 'meanstd':
            x_mean = np.mean(data, axis=axis, keepdims=True)  # 保持数据维度
            x_var = np.var(data, axis=axis, keepdims=True)
            return (data - x_mean) / (np.sqrt(x_var) + 1e-8)
        elif method == 'maxmin':
            x_max = np.max(data, axis=axis, keepdims=True)
            x_min = np.min(data, axis=axis, keepdims=True)
            return (data - x_min) / ((x_max - x_min) + 1e-8)

    def train(self, user):
        savePath = os.path.join(config.userInfoPath, user, 'data')
        srate = 500
        data, label = [], []
        for idx, sti in enumerate(self.sti_lst):
            data_files = glob(os.path.join(savePath, str(sti), '*'))
            for data_file in data_files:
                sub_data = loadmat(data_file)['data']
                # 限制为前8通道，排除可能的trigger标记段带来维度隐患
                sub_data = sub_data[:8, :]
                for i in range(sub_data.shape[-1] // srate):
                    data.append(sub_data[:, int(i * srate) : int((i + 1) * srate)])
                    label.append(idx)

        data = np.array(data)
        data = self.normalized(data, axis=-1)
        dataset = ConfuseDataset(data=(data, label))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1E-4)
        best_loss = 1e99999

        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()
            accu_loss = torch.zeros(1).to(self.device)  # 累计损失
            accu_num = torch.zeros(1).to(self.device)  # 累计预测正确的样本数
            sample_num = 0
            for step, data in enumerate(data_loader):
                eeg_norm, labels = data
                eeg_norm = eeg_norm.to(self.device).type(torch.float32)
                labels = labels.to(self.device).type(torch.int64)
                sample_num += len(labels)

                pred = self.model(eeg_norm)
                loss = criterion(pred, labels)

                loss.backward()

                pred_classes = torch.max(pred, dim=1)[1]
                accu_num += torch.eq(pred_classes, labels.to(self.device)).sum().detach()
                accu_loss += loss.detach()

                optimizer.step()
                optimizer.zero_grad()

            epoch_loss = (accu_loss / (step + 1)).detach().cpu().numpy()[0]
            epoch_acc = (accu_num / sample_num).detach().cpu().numpy()[0]

            if epoch_loss < best_loss:
                best_loss = epoch_loss

                save_folder = os.path.join(config.userInfoPath, user, 'weight')
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                torch.save(self.model.state_dict(), os.path.join(save_folder, f'best.pth'),
                           _use_new_zipfile_serialization=False)

            self.trainingProcessSignal.emit(round((epoch / self.epochs) * 100, 3))

        self.load_state_dict(user)

    def predict(self, data):
        with torch.no_grad():
            self.model.eval()
            
            # 使用模型设定好的通道数，从序列尾部截取真正的500长度，防止数据未对齐导致全连接层尺寸相乘异常以及第9脑电辅助通道引起计算图 squeeze 维度崩溃
            data = data[:8, -500:]
            
            data = self.normalized(data, axis=-1)

            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            out = self.model(data)

            out = torch.softmax(out, dim=1)
            out = out.detach().cpu().numpy()

            class_idx = np.argmax(out)
            class_prob = np.max(out, axis=-1)[0]

        return class_idx, class_prob


recognition = Recognition()

