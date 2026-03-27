#!/bin/bash

set -e

echo "====================================="
echo " Memory Server 依赖容器停止"
echo "====================================="

if [ ! -z "$(docker ps -q -f name=memory-qdrant)" ]; then
    echo "停止 Qdrant..."
    docker stop memory-qdrant
    docker rm memory-qdrant
    echo "✅ Qdrant 已停止"
fi

if [ ! -z "$(docker ps -q -f name=memory-paradedb)" ]; then
    echo "停止 ParadeDB..."
    docker stop memory-paradedb
    docker rm memory-paradedb
    echo "✅ ParadeDB 已停止"
fi
