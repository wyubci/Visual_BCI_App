import os
import yaml
import threading
from .dataNode.DataNode import DataNode

class Config(DataNode):
    def __init__(self, filename='config.yaml'):
        super().__init__()
        self.subjectName = 'TestSubject'
        self._filename = None

        self.userInfoPath = 'user'
        self.currentUser = None

        # ---------- SSVEP 刺激频率列表 (9 个: 8.0 ~ 15.8 Hz, ~1.0 Hz 间隔) ----------
        self.sti_lst = [
            8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 15.8,
        ]

        # ---------- SSVEP 相位列表 ----------
        self.sti_phase_lst = None  # 由代码自动生成棋盘格相位

        # ---------- SSVEP 网格参数 ----------
        self.ssvep_grid_rows = 3
        self.ssvep_grid_cols = 3
        self.ssvep_grid_target_count = 9    # 3×3=9 块
        self.ssvep_fixations_per_block = 1  # 每块 1 个注视点

        self.candidateStiList = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26]

        # ---- BrainVision LSL 设备配置 ----
        self.device_type = 'neuro_dance_tcp'           # 'lsl' | 'neuro_dance_serial' | 'neuro_dance_tcp'
        self.lsl_stream_name = 'BrainVision'          # LSL 流名称
        self.lsl_stream_type = 'EEG'                  # LSL 流类型
        self.lsl_selected_channels = [0, 1, 2, 3, 4, 5, 6, 7]  # NeuroDance 8 导
        self.lsl_target_sample_rate = 250             # 目标采样率 Hz

        # ---- 电极蒙太奇 (用于模拟不同密度配置) ----
        # 实际只有 8 导，保留此字段供未来扩展
        self.electrode_montage = {
            '8ch': [0, 1, 2, 3, 4, 5, 6, 7],
        }

        # M键模式下的移动距离设置
        self.m_mode_distances = {
            'forward': 20,   # 前进距离(厘米)
            'backward': 20,  # 后退距离(厘米)
            'left': 20,      # 左移距离(厘米)
            'right': 20      # 右移距离(厘米)
        }

        self.filename = filename

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, val):
        self._filename = val
        if os.path.isfile(self.filename):
            self.load()
        self.save()

    def load(self):
        # yaml_data = None
        # while yaml_data is None:
        with open(self.filename, 'r', encoding='utf8') as fp:
            yaml_data = yaml.load(fp, Loader=yaml.FullLoader)

        def _load(node, data):
            node_data = node.to_dict(False)
            for k, v in node_data.items():
                if k not in data:
                    continue
                if isinstance(v, DataNode):
                    new_yaml_data = data[k]
                    if not isinstance(new_yaml_data, dict):
                        raise Exception(f'{self.filename}字段{k}错误')
                    _load(v, new_yaml_data)
                elif isinstance(v, dict):
                    for k_, v_ in v.items():
                        if k_ not in data[k]:
                            continue
                        if isinstance(v_, DataNode):
                            new_yaml_data_ = data[k][k_]
                            if not isinstance(new_yaml_data_, dict):
                                raise Exception(f'{self.filename}字段{k}错误')
                            _load(v_, new_yaml_data_)
                else:
                    setattr(node, k, data[k])

        _load(self, yaml_data)

    def save(self):
        data = self.to_dict()
        temp_filename = self.filename + '.tmp'
        with open(temp_filename, 'w', encoding='utf8') as fp:
            yaml.dump(data=data,
                      stream=fp,
                      sort_keys=False,
                      encoding='utf8',
                      allow_unicode=True)
        if os.path.isfile(self.filename):
            os.remove(self.filename)
        os.rename(temp_filename, self.filename)

    def change(self, attr, val):
        if hasattr(self, attr):
            setattr(self, attr, val)
            self.save()


def get_config(config) -> Config:
    return Config(config)



BASE_DIR = os.path.dirname(os.path.dirname(__file__))

config = get_config(os.path.join(BASE_DIR, 'config.yaml'))

