"""
This script implements a video streaming server and client using OpenCV, NumPy, and socket libraries.
The server reads frames from a video file, encodes them in the h264 format, and sends them to the client.
The client receives the frames, decodes them, and displays the video stream in a window.

Usage:
- To install cv2, and loguru: pip install opencv-python loguru
- To run the server: python main.py --mode server --video_path video_1080p.mp4
- To run the client: python main.py --mode client
"""
import cv2  # 导入OpenCV库，用于处理图像和视频
import numpy as np  # 导入NumPy库，用于处理大型矩阵
import argparse
import socket  # 导入socket库，用于网络通信
import struct  # 导入struct库，用于处理C语言类型的数据
import sys  # 导入sys库，用于处理Python运行时环境的参数和函数
from loguru import logger  # 导入loguru库，用于日志记录


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

    def __init__(self, host, port, video_path):
        self.server_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )  # 创建一个TCP套接字
        self.server_socket.bind((host, port))  # 绑定到指定的主机和端口
        self.server_socket.listen(5)  # 开始监听连接
        self.video_path = video_path
        logger.info(f"服务器已启动，正在监听端口{port}...")

    def start_streaming(self):
        conn, addr = self.server_socket.accept()  # 接受一个连接
        logger.info(f"已连接到客户端{addr}。")

        cap = cv2.VideoCapture(self.video_path)  # 打开视频文件
        while cap.isOpened():  # 如果视频文件打开成功
            frames = []  # 创建一个列表来存储帧
            for _ in range(30):  # 每次读取30帧
                ret, frame = cap.read()  # 读取一帧
                if not ret:  # 如果读取失败
                    break  # 退出循环
                frames.append(frame)  # 将帧添加到列表

            for frame in frames:  # 对于每一帧
                encoded_frame = encode_frame(frame)  # 对帧进行编码

                if encoded_frame:  # 如果编码成功
                    # 发送帧长度和帧数据
                    conn.sendall(struct.pack("L", len(encoded_frame)) + encoded_frame)

            if not frames:  # 如果没有帧
                break  # 退出循环

        cap.release()  # 释放视频文件
        conn.close()  # 关闭连接


# 定义客户端类
class VideoStreamingClient:
    def __init__(self, host, port):
        self.client_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )  # 创建一个TCP套接字
        self.client_socket.connect((host, port))  # 连接到指定的主机和端口
        logger.info(f"已连接到服务器{host}:{port}。")

    def start_streaming(self):
        data = b""  # 创建一个空的字节字符串来存储数据
        payload_size = struct.calcsize("L")  # 计算“L”类型的大小
        while True:  # 无限循环
            while len(data) < payload_size:  # 如果数据的长度小于负载大小
                data += self.client_socket.recv(4096)  # 从套接字接收数据
            packed_msg_size = data[:payload_size]  # 获取打包的消息大小
            data = data[payload_size:]  # 获取剩余的数据
            msg_size = struct.unpack("L", packed_msg_size)[0]  # 解包消息大小

            while len(data) < msg_size:  # 如果数据的长度小于消息大小
                data += self.client_socket.recv(4096)  # 从套接字接收数据
            frame_data = data[:msg_size]  # 获取帧数据
            data = data[msg_size:]  # 获取剩余的数据

            # 解码并显示帧
            frame = cv2.imdecode(
                np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR
            )  # 解码帧数据
            cv2.imshow("Video", frame)  # 显示帧
            if cv2.waitKey(1) & 0xFF == ord("q"):  # 如果按下“q”键
                break  # 退出循环

        self.client_socket.close()  # 关闭套接字
        cv2.destroyAllWindows()  # 销毁所有窗口


if __name__ == "__main__":
    HOST, PORT = "localhost", 9999  # 定义主机和端口

    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Video Streaming Server and Client")

    # 添加参数
    parser.add_argument(
        "--mode", choices=["server", "client"], help="Mode: server or client"
    )
    parser.add_argument(
        "--video_path", default="video_1080p.mp4", help="Path to the video file"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 根据命令行参数运行服务器或客户端
    if args.mode == "server":
        server = VideoStreamingServer(HOST, PORT, args.video_path)
        server.start_streaming()
    elif args.mode == "client":
        client = VideoStreamingClient(HOST, PORT)
        client.start_streaming()
