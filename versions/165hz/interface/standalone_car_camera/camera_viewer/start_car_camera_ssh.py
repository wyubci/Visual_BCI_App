import os
import socket
import subprocess
import tempfile
import time

ROBOT_USER = "pi"
ROBOT_IP = "10.186.179.92"
LOCAL_PORT = 5001
REMOTE_PORT = 5000
PID_FILE = os.path.join(tempfile.gettempdir(), "standalone_car_camera_ssh.pid")
LAST_ATTEMPT_FILE = os.path.join(tempfile.gettempdir(), "standalone_car_camera_ssh.last")
RETRY_COOLDOWN_SECONDS = 25


def is_port_open(host="127.0.0.1", port=LOCAL_PORT, timeout=0.5):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _read_last_attempt():
    try:
        with open(LAST_ATTEMPT_FILE, "r", encoding="utf-8") as stamp_file:
            return float(stamp_file.read().strip())
    except (OSError, ValueError):
        return 0.0


def _write_last_attempt(ts):
    try:
        with open(LAST_ATTEMPT_FILE, "w", encoding="utf-8") as stamp_file:
            stamp_file.write(str(ts))
    except OSError:
        pass


def build_ssh_command_bat(bat_path):
    ssh_exe = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "OpenSSH", "ssh.exe")
    if not os.path.exists(ssh_exe):
        ssh_exe = "ssh"

    remote_cmd = "fuser -k 5000/tcp >/dev/null 2>&1 || killall python3 >/dev/null 2>&1; sleep 2; python3 ~/fast_stream.py"
    ssh_args = (
        f"-o ConnectTimeout=10 "
        f"-o ServerAliveInterval=30 "
        f"-o ServerAliveCountMax=3 "
        f"-o ExitOnForwardFailure=yes "
        f"-L {LOCAL_PORT}:127.0.0.1:{REMOTE_PORT} "
        f"-tt {ROBOT_USER}@{ROBOT_IP} \"{remote_cmd}\""
    )

    content = [
        "@echo off",
        "title Car Camera SSH Tunnel",
        "echo Starting SSH Tunnel to Car Camera...",
        f"echo connecting to {ROBOT_USER}@{ROBOT_IP}...",
        "echo.",
        "echo If this is the first time connecting, you may need to type yes to accept fingerprint.",
        "echo Please enter password if prompted.",
        "echo.",
        f"\"{ssh_exe}\" {ssh_args}",
        "if %ERRORLEVEL% NEQ 0 (",
        "    color 0C",
        "    echo.",
        "    echo SSH connection terminated with error code %ERRORLEVEL%.",
        "    echo Please check your network and Robot IP.",
        "    pause",
        ")",
    ]

    try:
        with open(bat_path, "w", encoding="cp936") as f:
            f.write("\n".join(content))
    except (UnicodeEncodeError, LookupError):
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))


def ensure_camera_tunnel_started():
    try:
        if is_port_open():
            return None

        now = time.time()
        last_attempt = _read_last_attempt()
        if now - last_attempt < RETRY_COOLDOWN_SECONDS:
            return None

        bat_file = os.path.join(tempfile.gettempdir(), "start_standalone_camera_tunnel.bat")
        build_ssh_command_bat(bat_file)
        _write_last_attempt(now)

        cmd_str = f'start "StandaloneCarCameraSSH" cmd /c "{bat_file}"'
        subprocess.Popen(cmd_str, shell=True)
        return True
    except Exception:
        return None


if __name__ == "__main__":
    ensure_camera_tunnel_started()
