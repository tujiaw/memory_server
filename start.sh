#!/bin/bash

set -e

echo "====================================="
echo " Memory Server 启动脚本"
echo "====================================="

if [ ! -f ".env" ]; then
    echo ".env 不存在，正在从 .env.example 创建..."
    cp .env.example .env
    echo "请先填写 .env 中的 OPENAI_API_KEY、SECRET_KEY 和 SERVICE_CLIENTS_JSON 后再重试。"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Docker 未安装，请先安装 Docker。"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "Python3 未安装。"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "安装/更新依赖..."
pip install -q -r requirements.txt

echo "启动依赖容器..."
docker compose up -d qdrant mongodb

echo "等待依赖服务就绪..."
sleep 3

echo "启动 API: http://localhost:8899"
echo "接口文档: http://localhost:8899/docs"
exec python main.py
