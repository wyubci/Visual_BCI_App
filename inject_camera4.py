import codecs
import re

file_path = r'interface/car_interface/car_window.py'
with codecs.open(file_path, 'r', 'utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if 'self.row1Layout.setSpacing(280)' in line:
        # Check if next line is addLayout
        pass

# Actually let's just rewrite everything
# Let's find "self.mainVLayout.addLayout(self.row1Layout)"
# If it's missing, add it after setSpacing(280)
out_lines = []
has_added = False
for line in lines:
    out_lines.append(line)
    if 'self.row1Layout.setSpacing(280)' in line:
        has_added = True
        out_lines.append("        self.mainVLayout.addLayout(self.row1Layout)\n")

# remove duplicates if there are multiple `self.mainVLayout.addLayout(self.row1Layout)`
with codecs.open(file_path, 'w', 'utf-8') as f:
    f.writelines(out_lines)
print("success")