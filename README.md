# Visual BCI App 脑控小车

> 开源项目：本仓库源码以 MIT License 发布。采集数据、模型权重、运行日志、缓存和虚拟环境不纳入仓库。

本仓库整理脑控小车/视觉 BCI 项目的主应用代码、SSVEP 刺激范式、离线评估脚本、小车视频流优化脚本和 165Hz 版本归档。

## 项目内容

| 内容 | 位置 | 说明 |
| --- | --- | --- |
| 主入口 | `main.py` | 启动 Visual BCI App |
| 脑控小车界面 | `interface/car_interface/` | SSVEP 刺激、FBCCA/TDCA 解码、小车控制接口和训练框架 |
| 设备连接 | `interface/deviceControl_interface/` | 设备状态、RDA/LSL 接收和信号显示 |
| 视频流工具 | `start_car_camera_ssh.py`、`update_car_code.py`、`interface/standalone_car_camera/` | 小车摄像头低延迟显示与部署 |
| 离线评估 | `benchmark_*.py`、`offline_*.py`、`run_*` | SSVEP/小车识别评估脚本 |
| 165Hz 归档 | `versions/165hz/` | 165Hz 刷新率版本源码归档 |

## 依赖

基础依赖见：

- `requirements.txt`
- `requirements.clean.txt`
- `requirements.local.txt`
- `environment/bci_env.yml`

建议新建独立 conda/venv 环境后安装依赖，不把本地环境目录提交到仓库。

## 运行

```powershell
python main.py
```

单独调试小车界面可参考：

```powershell
python interface/car_interface/car_window.py
```

## 上传边界

以下内容只保留在本地，不进入 GitHub：

- `saveCarData/`、`runtime_data/`、`ExperimentResults/`、`analysis_reports/`
- `.venv/`、`__pycache__/`、IDE 配置和缓存
- `*.mat`、`*.npz`、`*.npy`
- `*.pt`、`*.pth`、`*.joblib`、`*.pkl`、`*.onnx`
- `*.dat`、`*.whl`、`*.zip`、日志和临时文件

`shape_predictor_68_face_landmarks.dat`、本地训练权重和采集样本未上传；如运行时需要，请按项目内说明在本地放置。

## 参考文档

- `脑控小车实现总结.md`
- `SSVEP_CAR_FIX_README.md`
- `WINDOWS_ENCODING_README.md`
- `interface/standalone_car_camera/README.md`
