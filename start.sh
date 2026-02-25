#!/bin/bash

# Memory Server 本地启动脚本

set -e

echo "======================================"
echo "  Memory Server 本地启动脚本"
echo "======================================"

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "⚠️  .env 文件不存在，从 .env.example 复制..."
    cp .env.example .env
    echo "📝 请编辑 .env 文件设置 OPENAI_API_KEY"
    echo ""
    read -p "按 Enter 继续（确保已配置 .env）..."
fi

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查 Qdrant 容器
QDRANT_RUNNING=$(docker ps -q -f name=memory-qdrant)
if [ -z "$QDRANT_RUNNING" ]; then
    echo "🚀 启动 Qdrant..."
    docker run -d \
        --name memory-qdrant \
        --restart unless-stopped \
        -p 6333:6333 \
        -p 6334:6334 \
        -v qdrant_storage:/qdrant/storage \
        qdrant/qdrant:v1.12.0
    echo "✅ Qdrant 已启动"
    echo "   - Web UI: http://localhost:6333/dashboard"
    echo "   - API: http://localhost:6333"
else
    echo "✅ Qdrant 已在运行"
fi

# 检查 MongoDB 容器
MONGO_RUNNING=$(docker ps -q -f name=memory-mongodb)
if [ -z "$MONGO_RUNNING" ]; then
    echo "🚀 启动 MongoDB..."
    docker run -d \
        --name memory-mongodb \
        --restart unless-stopped \
        -p 27017:27017 \
        -v mongodb_data:/data/db \
        -e MONGO_INITDB_DATABASE=memory_server \
        mongo:8.0
    echo "✅ MongoDB 已启动"
    echo "   - 连接: mongodb://localhost:27017"
    echo "   - MongoDB Express: 可单独启动用于管理"
else
    echo "✅ MongoDB 已在运行"
fi

echo ""
echo "⏳ 等待数据库启动..."
sleep 3

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

echo "🔧 激活虚拟环境..."
source venv/bin/activate

echo "📦 安装/更新依赖..."
pip install -q -r requirements.txt

echo ""
echo "======================================"
echo "🚀 启动 Memory Server API..."
echo "======================================"
echo ""
echo "服务地址:"
echo "  - API: http://localhost:8899"
echo "  - 文档: http://localhost:8899/docs"
echo ""
echo "管理界面:"
echo "  - Qdrant: http://localhost:6333/dashboard"
echo "  - MongoDB: mongodb://localhost:27017"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动 API
python main.py
