import copy

class DataNode:

    def get_dict(self):
        return self.__dict__

    def to_dict(self, rec=True):
        data = {k: v for k, v in self.get_dict().items()
                if not k.startswith('_') and not k.startswith('m_') and not isinstance(v, classmethod)}
        if rec:
            d = copy.deepcopy(data)
            for k, v in d.items():
                if isinstance(v, DataNode):
                    d[k] = v.to_dict()
                if isinstance(v, dict):
                    for k_, v_ in v.items():
                        if isinstance(v_, DataNode):
                            d[k][k_] = v_.to_dict()
            return d
        else:
            return data