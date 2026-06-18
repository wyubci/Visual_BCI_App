adapter_code = '''
class TDCA_Opt:
    def __init__(self, Fs=250, ws=2, Nm=5, Nc=8, Nf=5, dataset="user", lagging_len=1, n_components=1):
        self.Fs = Fs
        self.ws = ws
        self.Nm = Nm
        self.Nc = Nc
        self.Nf = Nf
        self.dataset = dataset
        self.lagging_len = lagging_len
        self.n_components = n_components

from PyQt5.QtCore import QObject, pyqtSignal

class TDCA_Adapter(QObject):
    sendResultSignal = pyqtSignal(object)
    
    def __init__(self, num_harmonics, times, targets, Nh=8):
        super(TDCA_Adapter, self).__init__()
        self.Nh = Nh
        self.Fs = 250
        self.targets = targets
        
        opt = TDCA_Opt(Fs=self.Fs, ws=times-0.14, Nm=5, Nc=Nh, Nf=len(targets))
        self.tdca = TDCA(opt, targets)
        self.is_fitted = False
        
    def fit(self, X, y):
        self.tdca.fit(X, y)
        self.is_fitted = True
        
    def classify(self, test_data):
        if not self.is_fitted:
            import numpy as np
            # dummy fit just to allow running without crash if untrained
            dummy_X = np.random.randn(len(self.targets)*2, self.Nh, self.tdca.T)
            dummy_y = np.repeat(np.arange(len(self.targets)), 2)
            self.tdca.fit(dummy_X, dummy_y)
            self.is_fitted = True
            
        import numpy as np
        T = self.tdca.T
        test_data = test_data[:, -T:]
        X_test = test_data.reshape(1, self.Nh, T)
        
        sum_features = np.zeros((self.tdca.Nm, 1, self.tdca.Nf))
        FB_X_Test = self.tdca.filter_bank(X_test)
        for fb_i in range(self.tdca.Nm):
            fb_weight = (fb_i + 1) ** (-1.25) + 0.25
            sum_features[fb_i] = fb_weight * self.tdca.transform(FB_X_Test[fb_i], fb_i)
            
        sum_features = np.sum(sum_features, axis=0) # shape (1, Nf)
        pred_label = self.tdca.classes_[np.argmax(sum_features, axis=-1)][0]
        
        return pred_label
'''
with open('models/TDCA.py', 'a', encoding='utf-8') as f:
    f.write(adapter_code)
