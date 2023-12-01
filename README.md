# 项目介绍

这是一个关于使用python-opencv进行视频推流的项目。

# 安装指南

## 系统要求

- Python 3.x

## 安装步骤

1. 克隆项目到本地：

    ```bash
    git clone https://github.com/wzr2/video-pass
    ```

2. 进入项目目录：

    ```bash
    cd video-pass
    ```

3. 创建并激活虚拟环境（可选）：

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4. 安装依赖：

    ```bash
    pip install -r requirements.txt
    ```

# 使用方法

1. 放置视频

2. 启动服务器：

    ```bash
    bash start_server.sh
    ```

或者
    ```bash
    python core.py \
        --mode=server \
        --host=localhost \
        --port=9999 \
        --stati_dir=./results \
        --video_path=./video_1080p.mp4
    ```

3. 启动（多个）客户端

    ```bash
    bash start_multi_client.sh
    ```
    或者
    ```bash
    python main.py \
        --host=localhost \
        --port=9999 \
        --clients=8 \
        --stati_dir=./results
    ```