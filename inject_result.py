import codecs

file_path = r'interface/car_interface/car_window.py'
with codecs.open(file_path, 'r', 'utf-8') as f:
    text = f.read()

init_end_loc = '''        self.continuous_mode = False  # 连续刺激模式

        self._initLayout()'''

init_end_new = '''        self.continuous_mode = False  # 连续刺激模式
        
        # 创建本次实验结果保存文件夹
        current_time_str = time.strftime("%Y%m%d_%H%M%S")
        self.exp_result_dir = os.path.join("ExperimentResults", current_time_str)
        if not os.path.exists(self.exp_result_dir):
            os.makedirs(self.exp_result_dir)
        self.exp_txt_path = os.path.join(self.exp_result_dir, "results.txt")
        with open(self.exp_txt_path, "a", encoding="utf-8") as f:
            f.write(f"--- 脑控小车实验记录 ({current_time_str}) ---\\n")

        self._initLayout()'''

text = text.replace(init_end_loc, init_end_new)

set_result_loc = '''            print(f"=" * 50)
            print(f"识别结果: {command}")
            print(f"命令索引: {idx}")
            print(f"刺激频率: {self.sti_lst[idx]} Hz")
            print(f"=" * 50)

            # 发送命令到小车（通过Socket）'''

set_result_new = '''            print(f"=" * 50)
            print(f"识别结果: {command}")
            print(f"命令索引: {idx}")
            print(f"刺激频率: {self.sti_lst[idx]} Hz")
            print(f"=" * 50)
            
            # 追加保存至本地txt结果
            try:
                with open(self.exp_txt_path, "a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%H:%M:%S')}] 判定结果: {command}, 频率: {self.sti_lst[idx]}Hz, 对应指令索引: {idx}\\n")
            except Exception as e:
                print(f"无法写入结果至文本: {e}")

            # 发送命令到小车（通过Socket）'''

text = text.replace(set_result_loc, set_result_new)

with codecs.open(file_path, 'w', 'utf-8') as f:
    f.write(text)

print("success")
