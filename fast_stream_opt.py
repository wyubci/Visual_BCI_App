import cv2
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
