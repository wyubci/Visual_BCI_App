import socket
import time
import threading
import queue
# 密码：bci123456
class DroneController:
    def __init__(self, drone_address=('192.168.10.1', 8889)):
        self.drone_address = drone_address
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._is_running = True
        self.response_queue = queue.Queue()
        self.last_response = None
        self.waiting_for_response = False
        self._start_receiver()
        # self._start_battery_monitor()
        
    def _start_receiver(self):
        # 启动接收线程
        self.receiver_thread = threading.Thread(target=self._receiver)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
        
    def _receiver(self):
        # 接收无人机响应
        while self._is_running:
            try:
                response, _ = self.sock.recvfrom(1024)
                response_text = response.decode('utf-8').strip()
                
                # 保存最近的响应
                self.last_response = response_text
                
                # 将响应放入队列
                self.response_queue.put(response_text)
                
                # 检查响应是否是电量数据（纯数字）
                if response_text.isdigit():
                    # 电池电量格式
                    print(f"\n>>> 剩余电量：{response_text} <<<")
                else:
                    # 其他响应格式
                    print(f"\n>>> 无人机响应: {response_text} <<<")
                    
                # 重新显示输入提示
                print("请输入指令代码: ", end="", flush=True)
            except Exception as e:
                time.sleep(0.1)
    
    def wait_for_response(self, timeout=5):
        """等待无人机响应，超时返回None"""
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = self.response_queue.get(block=True, timeout=0.5)
                    return response
                except queue.Empty:
                    continue
            return None
        except Exception as e:
            print(f"等待响应出错: {e}")
            return None
                
    def send_command(self, command):
        # 发送命令到无人机
        print(f"发送命令: {command}")
        self.sock.sendto(command.encode('utf-8'), self.drone_address)
        
    def execute_action(self, action_code, distance=150):
        # 执行动作
        commands = {
            0: 'takeoff',
            1: 'land',
            2: f'up {50}',
            3: f'down {50}',
            4: f'forward {50}',
            5: f'back {50}',
            6: f'left {50}',
            7: f'right {50}',
        }
        
        if action_code in commands:
            # 清空响应队列
            while not self.response_queue.empty():
                self.response_queue.get()
                
            # 发送命令
            self.send_command(commands[action_code])
            
            # 如果是起飞命令，等待响应后再上升70厘米
            if action_code == 0:
                # 等待起飞命令的响应
                response = self.wait_for_response(timeout=10)
                if response == "ok":
                    print("起飞成功，开始上升60厘米")
                    # 执行上升70厘米
                    self.send_command(f'up 60')
                    print("执行组合动作: 起飞后上升46厘米")
                else:
                    print(f"起飞响应异常: {response}，取消上升动作")
        else:
            print(f"未知动作代码: {action_code}")
            
    def initialize(self):
        try:
            # 初始化无人机
            self.send_command('command')
            time.sleep(0.5)
            print("无人机连接成功")
        except Exception as e:
            print(f"无人机连接失败: {str(e)}，请检查连接")
        
    def close(self):
        # 关闭控制器
        self._is_running = False
        self.sock.close()
        
    def _start_battery_monitor(self):
        # 启动电池监控线程
        self.battery_thread = threading.Thread(target=self._battery_monitor)
        self.battery_thread.daemon = True
        self.battery_thread.start()
        
    def _battery_monitor(self):
        # 定期查询电池电量
        while self._is_running:
            self.send_command('battery?')
            time.sleep(5)  # 每5秒查询一次

def test_drone_control():
    # 无人机地址
    DRONE_ADDRESS = ('192.168.10.1', 8889)
    # 默认移动距离
    DEFAULT_DISTANCE = 150
    
    # 初始化无人机控制器
    drone = DroneController(drone_address=DRONE_ADDRESS)
    drone.initialize()
    
    # 需要设置距离的动作代码
    distance_actions = [2, 3, 4, 5, 6, 7, 10, 11, 12]
    
    try:
        print("\n" + "="*50)
        print("无人机控制系统已启动")
        print("可用指令: 起飞(0), 降落(1), 上升(2), 下降(3), 前进(4), 后退(5), 左移(6), 右移(7), 翻转(8), 电量(9), 右转(10), 左转(11), 速度(12), 退出(q)")












        print("注意: 起飞(0)命令会自动执行起飞后再上升70厘米")
        print("对于上升(2)、下降(3)、前进(4)、后退(5)、左移(6)、右移(7)、右转(10)、左转(11)，可以指定距离或角度，格式: 指令代码:值")
        print("对于速度(12)，可以指定飞行速度，格式: 12:值，例如: 12:50 表示设置速度为50厘米/秒")
        print("例如: 4:100 表示前进100厘米，10:90 表示右转90度")
        print("="*50 + "\n")
        
        while True:
            user_input = input("请输入指令代码: ")
            
            if user_input.lower() == 'q':
                print("退出控制系统")
                break
            
            # 检查是否包含距离设置（格式：指令代码:距离）
            if ':' in user_input:
                parts = user_input.split(':')
                if len(parts) == 2:
                    try:
                        action = int(parts[0])
                        distance = int(parts[1])
                        
                        if 0 <= action <= 12:
                            action_names = ["起飞", "降落", "上升", "下降", "前进", "后退", "左移", "右移", "前翻转", "电量", "右转", "左转", "速度"]
                            
                            if action in distance_actions:
                                if action == 10 or action == 11:
                                    print(f"执行动作: {action_names[action]}，角度: {distance}度")
                                elif action == 12:
                                    print(f"设置{action_names[action]}: {distance}厘米/秒")
                                else:
                                    print(f"执行动作: {action_names[action]}，距离: {distance}厘米")
                                drone.execute_action(action, distance=distance)
                            else:
                                print(f"执行动作: {action_names[action]}")
                                drone.execute_action(action)
                        else:
                            print("无效指令代码，请输入0-12之间的数字或q退出")
                    except ValueError:
                        print("请输入有效的指令代码和距离")
                else:
                    print("格式错误，正确格式为: 指令代码:距离")
            else:
                try:
                    action = int(user_input)
                    if 0 <= action <= 12:
                        action_names = ["起飞", "降落", "上升", "下降", "前进", "后退", "左移", "右移", "前翻转", "电量", "右转", "左转", "速度"]
                        
                        if action in distance_actions:
                            if action == 10 or action == 11:
                                print(f"执行动作: {action_names[action]}，角度: {DEFAULT_DISTANCE}度")
                            elif action == 12:
                                print(f"设置{action_names[action]}: {DEFAULT_DISTANCE}厘米/秒")
                            else:
                                print(f"执行动作: {action_names[action]}，距离: {DEFAULT_DISTANCE}厘米")
                            drone.execute_action(action, distance=DEFAULT_DISTANCE)
                        else:
                            print(f"执行动作: {action_names[action]}")
                            drone.execute_action(action)
                    else:
                        print("无效指令代码，请输入0-12之间的数字或q退出")
                except ValueError:
                    print("请输入有效的数字或q退出")
    finally:
        drone.close()
        
# 测试无人机运行
if __name__ == "__main__":
    test_drone_control()