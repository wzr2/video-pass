"""
This script implements a video streaming server and client using OpenCV, NumPy, and socket libraries.
The server reads frames from a video file, encodes them in the h264 format, and sends them to the client.
The client receives the frames, decodes them, and displays the video stream in a window.

Usage:
- To install cv2, and loguru: pip install opencv-python loguru
"""
import cv2  # 导入OpenCV库，用于处理图像和视频
import numpy as np  # 导入NumPy库，用于处理大型矩阵
import argparse
import socket  # 导入socket库，用于网络通信
import struct  # 导入struct库，用于处理C语言类型的数据
from loguru import logger  # 导入loguru库，用于日志记录
import time
import os
import pandas as pd
import threading

# 定义函数：将视频帧编码为h264格式
def encode_frame(frame):
    """
    对帧进行H264编码，并返回编码后的数据。

    Parameters:
        frame (numpy.ndarray): 输入的帧数据。

    Returns:
        bytes: 编码后的数据，如果编码失败则返回None。
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, encimg = cv2.imencode(".jpg", frame, encode_param)
    if result:
        return encimg.tobytes()
    else:
        return None

# 定义服务器类
class VideoStreamingServer:
    """
    服务器类，用于从视频文件中读取帧，对帧进行编码，然后将编码后的帧发送到客户端。
    """

    def __init__(self, host, port, video_path, stati_dir):
        self.server_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )  # 创建一个TCP套接字
        self.server_socket.bind((host, port))  # 绑定到指定的主机和端口
        self.server_socket.listen(5)  # 开始监听连接
        self.video_path = video_path
        self.stati_dir = stati_dir
        logger.info(f"服务器已启动，正在监听端口{port}...")

    def handle_client(self, conn, addr):
        thread_id = int(conn.recv(1024).decode("utf-8"))
        logger.info(f"新连接：{addr}, thread_id: {thread_id}")
        stati_save_to = os.path.join(self.stati_dir, f"server_{thread_id}.csv")
        
        cap = cv2.VideoCapture(self.video_path)  # 打开视频文件
        chunk_id = 0
        df = []
        try:
            while cap.isOpened():  # 如果视频文件打开成功
                chunk = b''
                encoding_time_start = time.time()
                for _ in range(30):  # 每次读取30帧
                    ret, frame = cap.read()  # 读取一帧
                    if not ret:  # 如果读取失败
                        break  # 退出循环
                    encoded_frame = encode_frame(frame)
                    if encoded_frame:
                        # 将每个帧的大小和数据添加到块中
                        chunk += struct.pack("L", len(encoded_frame)) + encoded_frame
                encoding_time_end = time.time()
                encoding_time = encoding_time_end - encoding_time_start

                t1 = time.time()
                conn.sendall(struct.pack("L", len(chunk)) + chunk)
                conn.recv(3)
                t4 = time.time()

                logger.info(f"Encoding time: {round(encoding_time, 3)} s")
                logger.info(f"Frame size: {round(len(chunk)/10**6, 3)} MB")
                logger.info(f"Transmission time: {round(t4-t1, 3)} s")
                df.append({'chunk_id': chunk_id, 't1': t1, 't4': t4, 'encoding_time': encoding_time, 'chunk_size': len(chunk)}) 

                chunk_id += 1
                if not chunk:
                    break
        except KeyboardInterrupt:
            logger.info("服务器已停止。")
        finally:
            # 将统计信息保存到单独的CSV文件中
            pd.DataFrame(df).to_csv(stati_save_to)
            cap.release()
            conn.close()
 
    def start_streaming(self):
        try:
            while True:
                # 接受新连接，accept 返回一个新的套接字和客户端地址
                client_socket, addr = self.server_socket.accept()
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket, addr))
                client_thread.start()
        except KeyboardInterrupt:
            logger.info("服务器已停止。")
        finally:
            self.server_socket.close()

# 定义客户端类
class VideoStreamingClient:
    def __init__(self, host, port, stati_save_to, thread_id=0):
        self.client_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )  # 创建一个TCP套接字
        self.stati_save_to = stati_save_to
        self.df = []
        self.thread_id = thread_id
        self.client_socket.connect((host, port))  # 连接到指定的主机和端口
        logger.info(f"已连接到服务器{host}:{port}。")

    def _recvall(self, sock, n):
        # 辅助函数来接收确切数量的数据
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def start_streaming(self):
        # send thread id to server
        self.client_socket.sendall(b"" + str(self.thread_id).encode("utf-8"))
        try:
            chunk_id = 0
            while True:
                # 接收整个块的大小
                chunk_size = struct.unpack("L", self._recvall(self.client_socket, struct.calcsize("L")))[0]
                t2 = time.time()
                chunk_data = self._recvall(self.client_socket, chunk_size)
                transmission_time = time.time() - t2
                chunk_size = len(chunk_data)
                # 分析块数据
                while chunk_data:
                    frame_size = struct.unpack("L", chunk_data[:struct.calcsize("L")])[0]
                    frame_data = chunk_data[struct.calcsize("L"):frame_size + struct.calcsize("L")]
                    chunk_data = chunk_data[frame_size + struct.calcsize("L"):]
                    
                    time_decode_start = time.time()
                    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                    time_decode_end = time.time()
                    decoding_time = time_decode_end - time_decode_start

                    self.client_socket.sendall(b"ACK")

                    # cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                self.df.append({'chunk_id': chunk_id, 't2': t2, 't3': t2+transmission_time, 'decoding_time': decoding_time, 'chunk_size': chunk_size})
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except:
            logger.info("客户端已停止。")
        finally:
            pd.DataFrame(self.df).to_csv(self.stati_save_to)
            cv2.destroyAllWindows()
            self.client_socket.close()

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Video Streaming Server and Client")

    # 添加参数
    parser.add_argument(
        "--mode", choices=["server", "client"], help="Mode: server or client"
    )
    parser.add_argument(
        "--video_path", default="video_1080p.mp4", help="Path to the video file"
    )
    parser.add_argument(
        "--stati_dir", default="./results"
    )
    parser.add_argument(
        "--host", default="localhost", help="Host"
    )
    parser.add_argument(
        "--port", default=9999, help="Port"
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 根据命令行参数运行服务器或客户端
    if args.mode == "server":
        server = VideoStreamingServer(args.host, args.port, args.video_path, stati_dir=args.stati_dir)
        server.start_streaming()
    elif args.mode == "client":
        client = VideoStreamingClient(args.host, args.port, stati_dir=args.stati_dir)
        client.start_streaming()
