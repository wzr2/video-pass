import argparse
import threading
from core import VideoStreamingClient
import os

def start_client(host, port, stati_path, thread_id):
    """
    创建并启动一个视频流客户端实例。
    """
    client = VideoStreamingClient(host, port, stati_path, thread_id)
    client.start_streaming()

def main(args):
    """
    启动多个客户端实例。
    """
    threads = []
    for i in range(args.clients):
        # 为每个客户端创建一个新的线程
        thread = threading.Thread(target=start_client, args=(args.host, args.port, os.path.join(args.stati_dir, f"client_{i}.csv"), i))
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Multi-Client Video Streaming")

    # 添加参数
    parser.add_argument("--host", default="localhost", help="Host address")
    parser.add_argument("--port", type=int, default=9999, help="Port number")
    parser.add_argument("--clients", type=int, default=1, help="Number of clients to start")
    parser.add_argument("--stati_dir", default="client_data", help="Base path for statistics file")

    # 解析命令行参数
    args = parser.parse_args()

    # 启动客户端
    main(args)
