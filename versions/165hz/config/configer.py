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

        self.sti_lst = [15.4,
                        8, 15.2, 8.2, 15, 8.4, 14.8, 8.6, 14.6, 8.8, 14.4,
                        9, 14.2, 9.2, 14, 9.4, 13.8, 9.6, 13.6, 9.8, 13.4,
                        10, 13.2, 10.2, 13, 10.4, 12.8, 10.6, 12.6, 10.8,
                        12.4, 11, 12.2, 11.2, 12, 11.4, 11.8,
                        11.6
                        ]

        # self.sti_lst = [15.4,
        #                 8, 8.8, 9.6, 10.4, 11.2, 12, 12.8, 13.6, 14.4, 15,
        #                 8.2, 9, 9.8, 10.6, 11.4, 12.2, 13, 13.8, 14.6, 15.2,
        #                 8.4, 9.2, 10, 10.8, 11.6, 12.4, 13.2, 14, 14.8,
        #                 8.6, 9.4, 10.2, 11, 11.8, 12.6, 13.4, 14.2]

        self.candidateStiList = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        
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

