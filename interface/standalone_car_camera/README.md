# Standalone Car Camera Module

This package extracts the car camera connection module from the project into a standalone runnable viewer.

## Files

- `camera_viewer/run_camera_viewer.py`: standalone desktop viewer entry
- `camera_viewer/car_video_panel.py`: TCP camera client + frame decode + UI panel
- `camera_viewer/start_car_camera_ssh.py`: auto-start SSH tunnel (Windows)
- `camera_viewer/fast_stream_opt.py`: camera stream server to run on robot (Raspberry Pi)
- `camera_viewer/requirements.txt`: Python dependencies
- `camera_viewer/start_viewer.bat`: one-click viewer launcher

## Quick Start (Windows client)

1. Install dependencies:

   ```bash
   pip install -r camera_viewer/requirements.txt
   ```

2. Start viewer:

   - Double-click `camera_viewer/start_viewer.bat`
   - or run:

   ```bash
   python camera_viewer/run_camera_viewer.py
   ```

## Robot side

Place `fast_stream_opt.py` on robot as `~/fast_stream.py`, then ensure SSH to robot works.
The viewer will try direct stream on `192.168.1.11:5000` first, then fallback to SSH tunnel `127.0.0.1:5001`.

## Config

Edit these constants if needed:

- `camera_viewer/car_video_panel.py`: `ROBOT_IP`
- `camera_viewer/start_car_camera_ssh.py`: `ROBOT_USER`, `ROBOT_IP`
