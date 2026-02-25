#!/bin/bash

echo "🛑 停止 Memory Server..."

# 停止 Qdrant
if [ ! -z "$(docker ps -q -f name=memory-qdrant)" ]; then
    echo "停止 Qdrant..."
    docker stop memory-qdrant
    docker rm memory-qdrant
    echo "✅ Qdrant 已停止"
fi

# 停止 MongoDB
if [ ! -z "$(docker ps -q -f name=memory-mongodb)" ]; then
    echo "停止 MongoDB..."
    docker stop memory-mongodb
    docker rm memory-mongodb
    echo "✅ MongoDB 已停止"
fi

echo "✅ 所有服务已停止"
