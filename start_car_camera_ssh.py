import os
import socket
import subprocess
import tempfile
import time


ROBOT_USER = 'pi'
ROBOT_IP = '192.168.1.11'
LOCAL_PORT = 5001
REMOTE_PORT = 5000
PID_FILE = os.path.join(tempfile.gettempdir(), 'visual_bci_car_camera_ssh.pid')
LAST_ATTEMPT_FILE = os.path.join(tempfile.gettempdir(), 'visual_bci_car_camera_ssh.last')
RETRY_COOLDOWN_SECONDS = 25


def is_port_open(host='127.0.0.1', port=LOCAL_PORT, timeout=0.5):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def build_ssh_command_bat(bat_path):
    ssh_exe = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'System32', 'OpenSSH', 'ssh.exe')
    if not os.path.exists(ssh_exe):
        ssh_exe = 'ssh'

    # To avoid "Address already in use" (Error 98) on the robot:
    # 1. Try to kill the specific process holding port 5000 using 'fuser'
    # 2. Fallback to 'killall python3' if fuser is missing
    # 3. Suppress permission errors (for system processes we can't kill)
    # 4. Wait 2 seconds for socket release (TIME_WAIT)
    remote_cmd = "fuser -k 5000/tcp >/dev/null 2>&1 || killall python3 >/dev/null 2>&1; sleep 2; python3 ~/fast_stream.py"

    ssh_args = (
        f'-o ConnectTimeout=10 '
        f'-o ServerAliveInterval=30 '
        f'-o ServerAliveCountMax=3 '
        f'-o ExitOnForwardFailure=yes '
        f'-L {LOCAL_PORT}:127.0.0.1:{REMOTE_PORT} '
        f'-tt {ROBOT_USER}@{ROBOT_IP} "{remote_cmd}"'
    )
    
    content = [
        '@echo off',
        'title Car Camera SSH Tunnel',
        'echo Starting SSH Tunnel to Car Camera...',
        f'echo connecting to {ROBOT_USER}@{ROBOT_IP}...',
        'echo.',
        'echo If this is the first time connecting, you may need to type "yes" to accept the fingerprint.',
        'echo Please enter password if prompted.',
        'echo.',
        f'"{ssh_exe}" {ssh_args}',
        'if %ERRORLEVEL% NEQ 0 (',
        '    color 0C',
        '    echo.',
        '    echo SSH connection terminated with error code %ERRORLEVEL%.',
        '    echo Please check your network connection and Robot IP.',
        '    pause',
        ')'
    ]
    
    try:
        with open(bat_path, 'w', encoding='cp936') as f: 
            f.write('\n'.join(content))
    except (UnicodeEncodeError, LookupError):
         with open(bat_path, 'w', encoding='utf-8') as f: 
            f.write('\n'.join(content))

    return bat_path


def _read_existing_pid():
    try:
        with open(PID_FILE, 'r', encoding='utf-8') as pid_file:
            return int(pid_file.read().strip())
    except (OSError, ValueError):
        return None


def _is_process_running(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _write_pid(pid):
    try:
        with open(PID_FILE, 'w', encoding='utf-8') as pid_file:
            pid_file.write(str(pid))
    except OSError:
        pass


def _read_last_attempt():
    try:
        with open(LAST_ATTEMPT_FILE, 'r', encoding='utf-8') as stamp_file:
            return float(stamp_file.read().strip())
    except (OSError, ValueError):
        return 0.0


def _write_last_attempt(ts):
    try:
        with open(LAST_ATTEMPT_FILE, 'w', encoding='utf-8') as stamp_file:
            stamp_file.write(str(ts))
    except OSError:
        pass


def _cleanup_stale_pid_file():
    pid = _read_existing_pid()
    if pid is not None and not _is_process_running(pid):
        try:
            os.remove(PID_FILE)
        except OSError:
            pass


def ensure_camera_tunnel_started():
    print("DEBUG: Checking tunnel status...")
    try:
        if is_port_open():
            print("DEBUG: Port 5001 is open. Tunnel already running.")
            return None

        # _cleanup_stale_pid_file() # PID tracking is less relevant with 'start' command as we lose the handle
        now = time.time()
        
        last_attempt = _read_last_attempt()
        if now - last_attempt < RETRY_COOLDOWN_SECONDS:
            print(f"DEBUG: Cooldown active. {RETRY_COOLDOWN_SECONDS - (now - last_attempt):.1f}s remaining.")
            return None

        # existing_pid = _read_existing_pid()
        # if _is_process_running(existing_pid):
        #     print(f"DEBUG: Process {existing_pid} is running.")
        #     return None

        bat_file = os.path.join(tempfile.gettempdir(), 'start_ssh_tunnel.bat')
        print(f"DEBUG: Creating batch file at {bat_file}")
        build_ssh_command_bat(bat_file)
        
        _write_last_attempt(now)
        print("DEBUG: Launching subprocess via 'start'...")
        
        # Use 'start' command to force a visible window. 
        cmd_str = f'start "CarCameraSSH" cmd /c "{bat_file}"'
        
        # shell=True is required for 'start' command
        subprocess.Popen(cmd_str, shell=True)
        
        print(f"DEBUG: Launch command executed: {cmd_str}")
        return True
    except Exception as e:
        print(f"Error starting tunnel: {e}")
        return None

if __name__ == '__main__':
    ensure_camera_tunnel_started()
