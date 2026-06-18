# This script helps the user upload the optimized fast_stream.py to the Raspberry Pi
# It uses the existing SSH credentials (via user interaction)

import os
import subprocess
import time

# Optimized server-side code (JPEG compression, Low latency)
OPTIMIZED_CODE = r'''import cv2
import socket
import struct
import pickle
import time

def main():
    # Setup Camera
    cap = cv2.VideoCapture(0)
    # Lower resolution for better latency/fps on WiFi
    cap.set(3, 320) # Width
    cap.set(4, 240) # Height
    cap.set(5, 30)  # FPS
    
    # Try MJPG if camera supports it (faster than YUYV)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # CRITICAL: Disable Nagle's algorithm for low latency
    server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        server_socket.bind(('0.0.0.0', 5000))
        server_socket.listen(1)
        print("Optimized Camera Stream Ready. Listening on 5000...")
        
        while True:
            print("Waiting for client...")
            client_socket, addr = server_socket.accept()
            print("Client connected:", addr)
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # JPEG Quality 50 is a good balance for speed/size
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("Start failed")
                        break
                    
                    # COMPRESS TO JPEG
                    # This reduces data size from ~230KB (320x240x3) to ~10-20KB
                    success, encimg = cv2.imencode('.jpg', frame, encode_param)
                    
                    if success:
                        # Pickle the COMPRESSED buffer
                        # We send it as a numpy array, client will detect and decode
                        data = pickle.dumps(encimg)
                        
                        # Send header + data
                        client_socket.sendall(struct.pack("Q", len(data)) + data)
                        
            except Exception as e:
                print("Stream error:", e)
            finally:
                client_socket.close()
                print("Connection closed")
                
    except Exception as e:
        print("Bind error:", e)

if __name__ == '__main__':
    main()
'''

def create_update_script():
    # Save the python code to a local temp file
    with open("fast_stream_opt.py", "w", encoding='utf-8') as f:
        f.write(OPTIMIZED_CODE)
    
    print("Created local optimized script: fast_stream_opt.py")
    print("Preparing to upload to Raspberry Pi...")
    print("Please follow the prompts in the new window.")
    
    # Create a batch file to handle the SCP upload with user interaction
    # We rename it to fast_stream.py on the destination
    cmd = 'scp -o ConnectTimeout=10 -o StrictHostKeyChecking=no fast_stream_opt.py pi@192.168.1.11:~/fast_stream.py'
    
    bat_content = [
        '@echo off',
        'title Uploading Optimized Car Code',
        'echo ===================================================',
        'echo  UPLOADING OPTIMIZED CAMERA CODE TO ROBOT',
        'echo ===================================================',
        'echo.',
        'echo Connection: pi@192.168.1.11',
        'echo Target: ~/fast_stream.py',
        'echo.',
        'echo Please enter the robot password when prompted below:',
        'echo.',
        f'{cmd}',
        'if %ERRORLEVEL% EQU 0 (',
        '    color 0A',
        '    echo.',
        '    echo [SUCCESS] Code updated successfully!',
        '    echo You can now restart the main application.',
        ') else (',
        '    color 0C',
        '    echo.',
        '    echo [ERROR] Upload failed. Check password or connection.',
        ')',
        'echo.',
        'pause'
    ]
    
    with open("upload_code.bat", "w", encoding='cp936') as f:
        f.write('\n'.join(bat_content))
        
    # Run it
    os.system('start cmd /c upload_code.bat')

if __name__ == '__main__':
    create_update_script()
