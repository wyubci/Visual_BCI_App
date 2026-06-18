
clear;
clc;
test_datas = load('user/哈哈哈/data/8.8/1.mat').data;
test_data11=test_datas(2:8,:);%取出第一组数据
test_data11 = mean(test_datas);
% plot(test_data11(1:500));
% ylim([-10 10])

Fs=250;% 采样频率
T=1/Fs;% 采样时间
L=1500;% 信号长度
t = (0:L-1)*T; % 时间

y =test_data11;%信号
% figure;
% plot(t,y)
% title('信号')
% xlabel('时间(s)')
N = 2^nextpow2(L); %采样点数，采样点数越大，分辨的频率越精确，N>=L，超出的部分信号补为0
Y = fft(y,N)/N*2; %除以N乘以2才是真实幅值，N越大，幅值精度越高
f = Fs/N*(0:1:N-1); %频率
A = abs(Y); %幅值
P = angle(Y); %相值
figure;
plot(f(1:N/2),A(1:N/2)); %函数fft返回值的数据结构具有对称性,因此我们只取前一半
title(['幅值频谱',num2str(i)])
xlabel('频率(Hz)')
ylabel('幅值')
xlim([0 70])
ylim([0 1])