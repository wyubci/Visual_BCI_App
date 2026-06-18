import codecs

file_path = r'interface/car_interface/car_window.py'
with codecs.open(file_path, 'r', 'utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if 'self.continuous_mode = False' in line and '连续刺激模式' in line:
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(indent + "current_time_str = time.strftime('%Y%m%d_%H%M%S')\n")
        new_lines.append(indent + "self.exp_result_dir = os.path.join('ExperimentResults', current_time_str)\n")
        new_lines.append(indent + "if not os.path.exists(self.exp_result_dir):\n")
        new_lines.append(indent + "    os.makedirs(self.exp_result_dir)\n")
        new_lines.append(indent + "self.exp_txt_path = os.path.join(self.exp_result_dir, 'results.txt')\n")
        new_lines.append(indent + "try:\n")
        new_lines.append(indent + "    with open(self.exp_txt_path, 'a', encoding='utf-8') as f: f.write(f'--- 脑控小车实验记录 ({current_time_str}) ---\\n')\n")
        new_lines.append(indent + "except Exception: pass\n")
        
    if 'print(f"刺激频率: {self.sti_lst[idx]} Hz")' in line:
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(indent + 'try:\n')
        new_lines.append(indent + '    with open(self.exp_txt_path, "a", encoding="utf-8") as f:\n')
        new_lines.append(indent + '        f.write(f"[{time.strftime(\'%H:%M:%S\')}] 判定结果: {command}, 频率: {self.sti_lst[idx]}Hz, 指令索引: {idx}\\n")\n')
        new_lines.append(indent + 'except Exception: pass\n')

with codecs.open(file_path, 'w', 'utf-8') as f:
    f.writelines(new_lines)
print("success")
