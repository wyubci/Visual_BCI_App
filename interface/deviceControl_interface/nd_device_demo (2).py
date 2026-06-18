import math
import time

from neuro_dance.nd_device_process import NdDeviceBase
from neuro_dance.linkedlist import NdLinkedlist
import numpy as np

# 每秒钟接收的数据包数量
package_per_second = 5
# 存储5分钟的数据，计算存储的包数量
eeg_package_count = package_per_second * 60 * 5  # 脑电图数据包数量
eog_package_count = package_per_second * 60 * 5  # 眼电图数据包数量


class NdDevice(NdDeviceBase):
    """
    神经接口设备类，用于与硬件设备通信并处理脑电图和眼电图数据
    继承自NdDeviceBase基类
    """
    # 使用链表存储脑电图和眼电图数据
    eeg_datas = NdLinkedlist()  # 存储脑电图数据的链表
    eog_datas = NdLinkedlist()  # 存储眼电图数据的链表
    eeg_sample = 1000  # 脑电图采样率默认为1000Hz
    mode = None  # 设备通信模式
    last_read_timestamp = 0  # 初始化为0
    
    def __init__(self, mode, com, tcp_ip, tcp_port, host_mac_bytes=None):
        """
        初始化设备
        :param mode: 通信模式，'serial'为串口模式，'tcp'为网络模式
        :param com: 串口号，串口模式下使用
        :param tcp_ip: TCP服务器IP地址，网络模式下使用
        :param tcp_port: TCP服务器端口，网络模式下使用
        :param host_mac_bytes: 主机MAC地址字节数组，可选
        """
        #super(NdDeviceBase, self).__init__(mode, com, tcp_ip, tcp_port, host_mac_bytes)
        NdDeviceBase.__init__(self, mode, com, tcp_ip, tcp_port, host_mac_bytes)
        self.mode = mode

    def host_info(self):
        """
        获取主机(dongle)信息，包括版本、序列号和MAC地址
        """
        super(NdDevice, self).host_version_info()
        time.sleep(0.01)  # 短暂延时，避免通信冲突
        super(NdDevice, self).host_sn_info()
        time.sleep(0.01)
        super(NdDevice, self).host_mac_info()

    def device_info(self):
        """
        获取设备信息，包括版本、序列号和MAC地址
        """
        super(NdDevice, self).device_version_info()
        time.sleep(0.01)  # 短暂延时，避免通信冲突
        super(NdDevice, self).device_sn_info()
        time.sleep(0.01)
        super(NdDevice, self).device_mac_info()

    def battery(self):
        """
        获取设备电池电量信息
        """
        super(NdDevice, self).device_battery()

    def device_pair(self, mac):
        """
        与指定MAC地址的设备配对
        :param mac: 设备MAC地址
        """
        super(NdDevice, self).pair(mac)

    # 扫描蓝牙设备
    # 扫描到的设备通过devices_received(devices)回调函数返回
    def devices_scan(self):
        """
        扫描可用的蓝牙设备
        扫描结果通过devices_received回调函数返回
        """
        super(NdDevice, self).device_scan()

    def host_mac_received(self, host_mac):
        """
        接收到主机MAC地址的回调函数
        :param host_mac: 主机MAC地址
        """
        print("Test host_mac_received:"+host_mac)

    def host_sn_received(self, host_sn):
        """
        接收到主机序列号的回调函数
        :param host_sn: 主机序列号
        """
        print("Test host_sn_received:"+host_sn)

    def channel_received(self, data):
        """
        接收到通道数据的回调函数
        :param data: 通道数据
        """
        print("Test channel_received:"+data)

    def host_version_received(self, host_version):
        """
        接收到主机版本信息的回调函数
        :param host_version: 主机版本信息
        """
        print("Test host_version_received:"+host_version)

    def device_mac_received(self, device_mac):
        """
        接收到设备MAC地址的回调函数
        :param device_mac: 设备MAC地址
        """
        print("Test device mac:{0}".format(device_mac))

    def device_sn_received(self, device_sn):
        """
        接收到设备序列号的回调函数
        :param device_sn: 设备序列号
        """
        print("Test device_sn_received:{0}".format(device_sn))

    def device_battery_received(self, battery):
        """
        接收到设备电池电量信息的回调函数
        :param battery: 电池电量
        """
        print("Test battery:{0}".format(battery))

    def device_version_received(self, device_version):
        """
        接收到设备版本信息的回调函数
        :param device_version: 设备版本信息
        """
        print("Test device version:{0}".format(device_version))

    def eeg_received(self, data):
        """
        接收到脑电图数据的回调函数
        :param data: 包含时间戳和数据的字典
        """
        shape = self.array_shape(data['data'])
        # print("unix milliseconds(first point time):{0},shape:{1},data:{2}".format(data['timestamp'], shape, data['data']))
        # 如果数据超过设定的存储量，移除最早的数据
        if self.eeg_datas.length() > eeg_package_count:
            self.eeg_datas.removeHead()
        self.eeg_datas.add(data)  # 将新数据添加到链表中

    def eog_received(self, data):
        """
        接收到眼电图数据的回调函数
        :param data: 包含时间戳和数据的字典
        """
        shape = self.array_shape(data['data'])
        # print("unix milliseconds(first point time):{0},shape:{1},data:{2}".format(data['timestamp'], shape, data['data']))
        # 如果数据超过设定的存储量，移除最早的数据
        if self.eog_datas.length() > eog_package_count:
            self.eog_datas.removeHead()
        self.eog_datas.add(data)  # 将新数据添加到链表中

    def read_latest_eeg_data(self, target_freq=250):
        """
        读取最新的脑电图数据，并按照目标频率进行下采样
        :param target_freq: 目标采样频率（Hz），默认250Hz
        :return: 下采样后的脑电图数据，形状为(8,N)
        """
        current_time = int(round(time.time() * 1000))
        
        # 设备实际采样率
        device_freq = 1000  # 设备采样率1000Hz
        
        # 计算下采样比率
        downsample_ratio = int(device_freq / target_freq)  # 1000/250 = 4
        
        # 获取最新数据包
        latest_packet = None
        latest_time = 0
        
        for i in range(self.eeg_datas.length()):
            packet = self.eeg_datas.eleAt(i)
            if packet is None:
                continue
            
            if packet['timestamp'] > latest_time:
                latest_time = packet['timestamp']
                latest_packet = packet
        
        # 如果找到新数据包且比上次读取的更新
        if latest_packet and latest_time > self.last_read_timestamp:
            self.last_read_timestamp = latest_time
            
            # 获取原始数据
            raw_data = latest_packet['data']
            
            # 如果是numpy数组，进行下采样
            if isinstance(raw_data, np.ndarray):
                # 对每个通道的数据进行下采样（每downsample_ratio个点取1个）
                downsampled_data = raw_data[:, ::downsample_ratio]
                return downsampled_data
            else:
                # 对列表形式的数据进行下采样
                downsampled_data = []
                for channel in raw_data:
                    downsampled_channel = channel[::downsample_ratio]
                    downsampled_data.append(downsampled_channel)
                return np.array(downsampled_data)
        
        return None

    def __read_eeg_date_tcp(self, start_millis_second, read_millisecond, freq):
        """
        TCP模式下读取指定时间段的脑电图数据
        :param start_millis_second: 起始时间戳（毫秒）
        :param read_millisecond: 要读取的时间长度（毫秒）
        :param freq: 采样频率（Hz）
        :return: 指定时间段的脑电图数据，如果数据不足则返回None
        """
        point_per_millis = freq / 1000  # 每毫秒的数据点数
        eeg_data = None
        while self.eeg_datas.length() > 0:
            eeg_left = self.eeg_datas.eleAt(0)  # 获取链表中第一个数据包
            if eeg_left is None:
                break
            packet_start_millis = eeg_left['timestamp']  # 数据包的起始时间戳
            packet_end_millis = len(eeg_left['data'][0]) * 1000 / freq + packet_start_millis  # 数据包的结束时间戳
            # 如果数据包结束时间早于请求的起始时间，则移除该数据包
            if packet_end_millis < start_millis_second:
                self.eeg_datas.removeHead()
                continue
            # 计算数据起始位置
            eeg_start_position = int((start_millis_second - packet_start_millis) * point_per_millis)
            if eeg_start_position < 0:
                eeg_start_position = 0
            # 多加10个冗余点，确保数据足够
            need_point_count = int(read_millisecond * point_per_millis)
            check_point_count = need_point_count + eeg_start_position + 10
            # 检查是否有足够的数据点
            enough, channel_data = self.__check_data_enough(self.eeg_datas, check_point_count)
            if enough:
                # 将多个通道的数据水平拼接
                eeg_data = np.hstack(channel_data)
                # 截取指定时间段的数据
                eeg_data = eeg_data[:, eeg_start_position:(eeg_start_position + need_point_count)]
                break
        return eeg_data

    def __check_data_enough(self, datas, read_point_count):
        """
        检查是否有足够的数据点
        :param datas: 数据链表
        :param read_point_count: 需要的数据点数
        :return: (是否足够, 数据列表)
        """
        eleIndex = 0
        count = 0
        need_data = []
        # 遍历链表，累计数据点数
        while datas.length() > eleIndex and count < read_point_count:
            ele = datas.eleAt(eleIndex)
            if ele is None:
                break
            count = count + len(ele['data'][0])  # 累加数据点数
            eleIndex = eleIndex + 1
            need_data.append(ele['data'])  # 收集数据
        return count >= read_point_count, need_data  # 返回是否有足够的数据点和收集的数据

    def read_eeg_data(self, start_millis_second, read_millisecond, freq = 250):
        """
        读取指定时间段的脑电图数据，根据模式选择对应的读取方法
        :param start_millis_second: 起始时间戳（毫秒）
        :param read_millisecond: 要读取的时间长度（毫秒），注意不是结束时间戳！
        :param freq: 采样频率（Hz），默认250Hz
        :return: 指定时间段的脑电图数据
        """
        if self.mode == 'serial':
            return self.__read_eeg_from_serial(start_millis_second, read_millisecond, freq)
        elif self.mode == 'tcp':
            return self.__read_eeg_date_tcp(start_millis_second, read_millisecond, freq)

    def __read_eeg_from_serial(self, start_millis_second, read_millisecond, freq = 1000):
        """
        串口模式下读取指定时间段的脑电图数据
        :param start_millis_second: 起始时间戳（毫秒）
        :param read_millisecond: 要读取的时间长度（毫秒）
        :param freq: 采样频率（Hz），默认1000Hz
        :return: 指定时间段的脑电图数据，如果数据不足则返回None
        """
        packet_size = freq / package_per_second  # 每个数据包的点数
        packet_time = 200  # 每个数据包的时间长度（毫秒）
        point_per_millis = packet_size / packet_time  # 每毫秒的数据点数
        # 以防万一，多取两个包，保证截取时足够长
        packet_num = math.ceil(read_millisecond * point_per_millis / packet_size) + 2
        eeg_data = None
        while self.eeg_datas.length() > 0:
            eeg_left = self.eeg_datas.eleAt(0)  # 获取链表中第一个数据包
            if eeg_left is None:
                break
            eeg_packet_start_millis = eeg_left['timestamp']  # 数据包的起始时间戳
            # 如果数据包结束时间早于请求的起始时间，则移除该数据包
            if eeg_packet_start_millis + packet_time < start_millis_second:
                self.eeg_datas.removeHead()
                continue
            # 确保有足够的数据包
            if self.eeg_datas.length() > packet_num:
                # 计算数据起始位置
                eeg_start_position = int((start_millis_second - eeg_packet_start_millis) * point_per_millis)
                if eeg_start_position < 0:
                    print("eeg time error:{0}".format(eeg_start_position))
                    eeg_start_position = 0
                need_points = int(read_millisecond * point_per_millis)  # 需要的数据点数
                eeg_tmp = []
                # 收集足够的数据包
                for i in range(packet_num):
                    chan = self.eeg_datas.eleAt(i)
                    eeg_tmp.append(chan['data'])
                # 将多个通道的数据水平拼接
                eeg_data = np.hstack(eeg_tmp)
                # 截取指定时间段的数据
                eeg_data = eeg_data[:, eeg_start_position:(eeg_start_position + need_points)]
                break
        return eeg_data

    def devices_received(self, devices):
        """
        接收到设备列表的回调函数
        :param devices: 设备列表
        """
        print("devices count:{0}".format(len(devices)))
        for device in devices:
            print("name:{0},mac:{1},rssi:{2}".format(device['name'], device['mac'], device['rssi']))

    def cmd_error(self, cmd, pm, code):
        """
        命令错误的回调函数
        :param cmd: 命令
        :param pm: 参数
        :param code: 错误代码
        """
        print("cmd error:{0},pm:{1},code:{2}".format(cmd, pm, code))

    def crc_error(self):
        """
        CRC校验错误的回调函数
        """
        print("crc error")

    def array_shape(self, arr):
        """
        递归获取数组的形状
        :param arr: 数组
        :return: 数组的形状
        """
        if isinstance(arr, list):
            return [len(arr)] + self.array_shape(arr[0])
        else:
            return []

def tcp_test():
    """
    TCP模式测试函数
    通过TCP连接获取脑电图数据
    """
    # 创建TCP模式的设备实例
    nd_device = NdDevice(mode='tcp', com='', tcp_ip='192.168.0.118', tcp_port=8899, host_mac_bytes=None)
    nd_device.start()  # 启动设备
    
    # 数据质量统计变量
    total_packets = 0  # 成功接收的数据包总数
    no_data_count = 0  # 无数据次数
    test_duration = 10  # 测试持续时间（秒）
    expected_packets = test_duration * 5  # 理论上应该接收的数据包数（每秒5个）
    
    packet_timestamps = []  # 记录每个数据包的时间戳
    packet_shapes = []  # 记录每个数据包的形状
    start_time = time.time()  # 记录开始时间
    
    print(f"\n========== 开始数据质量测试 ==========")
    print(f"测试时长: {test_duration}秒")
    print(f"理论数据包数: {expected_packets}个 (每秒5个)")
    print(f"读取间隔: 0.1秒")
    print("=" * 50 + "\n")
    
    index = 0
    while time.time() - start_time < test_duration:  # 运行指定的测试时长
        time.sleep(0.1)  # 每0.1秒读取一次数据

        index = index + 1
        current_time = time.time() - start_time  # 当前经过的时间
        millis_second = int(round(time.time() * 1000))  # 获取当前时间戳（毫秒）
        
        read_data = nd_device.read_latest_eeg_data()  # 读取数据
        
        # 处理数据格式 - 从(8, N, 1)格式转换为(8, N)
        if read_data is not None and len(read_data.shape) == 3 and read_data.shape[2] == 1:
            read_data = read_data.reshape(read_data.shape[0], read_data.shape[1])
        
        # 统计数据
        if read_data is not None:
            total_packets += 1
            packet_timestamps.append(current_time)
            packet_shapes.append(read_data.shape)
            print(f"[{current_time:.2f}s] 数据包 #{total_packets}: shape={read_data.shape}")
            print(f"数据统计: 最小值={np.min(read_data):.2f}, "f"最大值={np.max(read_data):.2f}, "f"平均值={np.mean(read_data):.2f}")
        else:
            no_data_count += 1
            print(f"[{current_time:.2f}s] No data (第{no_data_count}次)")
    
    # 测试结束，打印统计结果
    print("\n" + "=" * 50)
    print(f"========== 数据质量测试结果 ==========")
    print("=" * 50)
    print(f"实际测试时长: {time.time() - start_time:.2f}秒")
    print(f"读取次数: {index}次")
    print(f"成功接收数据包: {total_packets}个")
    print(f"无数据次数: {no_data_count}次")
    print(f"理论数据包数: {expected_packets}个")
    
    # 计算丢包率
    if expected_packets > 0:
        packet_loss_rate = ((expected_packets - total_packets) / expected_packets) * 100
        print(f"丢包率: {packet_loss_rate:.2f}%")
    
    # 计算数据接收率
    if index > 0:
        success_rate = (total_packets / index) * 100
        print(f"数据接收成功率: {success_rate:.2f}%")
    
    # 分析数据包间隔
    if len(packet_timestamps) > 1:
        intervals = [packet_timestamps[i] - packet_timestamps[i-1] 
                    for i in range(1, len(packet_timestamps))]
        avg_interval = np.mean(intervals)
        min_interval = np.min(intervals)
        max_interval = np.max(intervals)
        std_interval = np.std(intervals)
        
        print("\n数据包接收间隔统计:")
        print(f"  平均间隔: {avg_interval:.3f}秒")
        print(f"  最小间隔: {min_interval:.3f}秒")
        print(f"  最大间隔: {max_interval:.3f}秒")
        print(f"  标准差: {std_interval:.3f}秒")
        print(f"  理论间隔: 0.200秒 (每秒5个包)")
    
    # 检查数据包形状一致性
    if packet_shapes:
        unique_shapes = list(set(packet_shapes))
        print(f"\n数据包形状统计:")
        for shape in unique_shapes:
            count = packet_shapes.count(shape)
            print(f"  {shape}: {count}个 ({count/len(packet_shapes)*100:.1f}%)")
    
    print("\n链表中存储的数据包总数:", nd_device.eeg_datas.length())
    print("=" * 50 + "\n")

    nd_device.close()  # 关闭设备连接

def serial_test():
    """
    串口模式测试函数
    通过串口连接获取脑电图数据
    """
    # 创建串口模式的设备实例
    nd_device = NdDevice(mode='serial', com='com5', tcp_ip='', tcp_port=None, host_mac_bytes='CD87113D47F1')
    nd_device.start()  # 启动设备
    nd_device.host_info()  # 获取主机信息
    nd_device.device_info()  # 获取设备信息
    nd_device.host_device_connect()  # 连接主机和设备
    nd_device.battery()  # 获取电池信息
    # 以下是眼电图配置，默认被注释掉
    # nd_device.eog_channel_config(1000)
    # nd_device.eog_channel_enable()
    # nd_device.eeg_disable()
    
    # 配置脑电图通道，采样率为1000Hz
    nd_device.eeg_channel_config(1000)
    nd_device.eeg_channel_enable()  # 启用脑电图通道
    
    index = 0
    while index < 10000:  # 循环10000次
        time.sleep(0.1)  # 每0.1秒读取一次数据

        index = index + 1
        millis_second = int(round(time.time() * 1000))  # 获取当前时间戳（毫秒）
        time_span = 1000  # 读取过去1秒的数据
        read_data = nd_device.read_eeg_data(millis_second - time_span, time_span)  # 读取数据
        # 如果需要打印数据和连接状态，可以取消下面的注释
        if read_data is not None:
            print(read_data)
            print("Device Connect:{0}".format(nd_device.is_device_connected()))
    
    # 测试完成后，禁用通道并关闭连接
    nd_device.eeg_disable()  # 禁用脑电图通道
    nd_device.eog_disable()  # 禁用眼电图通道
    nd_device.close()  # 关闭设备连接

# 查看PyCharm帮助：https://www.jetbrains.com/help/pycharm/


if __name__ == '__main__':
    tcp_test()  # 运行TCP模式测试