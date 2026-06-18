import codecs
import re

file_path = r'interface/car_interface/car_window.py'
with codecs.open(file_path, 'r', 'utf-8') as f:
    text = f.read()

# Fix the missing `self.mainVLayout.addLayout(self.row1Layout)`
broken_part = '''        self.row1Layout.setSpacing(280)  # 增加前进和后退之间的水平距离


        # 中间监控布局 - Astra 摄像头'''

fixed_part = '''        self.row1Layout.setSpacing(280)  # 增加前进和后退之间的水平距离
        self.mainVLayout.addLayout(self.row1Layout)

        # 中间监控布局 - Astra 摄像头'''

text = text.replace(broken_part, fixed_part)

with codecs.open(file_path, 'w', 'utf-8') as f:
    f.write(text)

print("success")