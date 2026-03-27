#!/bin/bash

set -e

# 默认使用阿里云 PyPI 镜像；可 export PIP_INDEX_URL=... 覆盖
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"

echo "====================================="
echo " Memory Server 启动脚本"
echo "====================================="

if [ ! -f ".env" ]; then
    echo ".env 不存在，正在从 .env.example 创建..."
    cp .env.example .env
    echo "请先填写 .env 中的 DATABASE_URL、OPENAI_API_KEY、SECRET_KEY 和 SERVICE_CLIENTS_JSON 后再重试。"
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

echo "安装/更新依赖... (pip: ${PIP_INDEX_URL})"
pip install -q -r requirements.txt \
  -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" \
  --default-timeout=120

echo "启动依赖容器..."
# 与 docker-compose 中 ${DATA_DIR:-./data} 对齐；可在 .env 写 DATA_DIR=/你的挂载盘路径
DR="${DATA_DIR:-}"
if [ -z "$DR" ] && [ -f .env ]; then
  DR=$(grep -E '^[[:space:]]*DATA_DIR=' .env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '\r' | sed "s/^['\"]//;s/['\"]$//")
fi
DR="${DR:-./data}"
mkdir -p "$DR/qdrant" "$DR/paradedb"
export DATA_DIR="$DR"
docker-compose up -d qdrant paradedb

echo "等待依赖服务就绪..."
sleep 3

echo "启动 API: http://localhost:8899"
echo "接口文档: http://localhost:8899/docs"
exec python main.py
