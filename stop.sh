#!/bin/bash

set -e

echo "====================================="
echo " Memory Server 依赖容器停止"
echo "====================================="

for c in memory-elasticsearch memory-postgres memory-api; do
  if [ -n "$(docker ps -q -f name=^/${c}$)" ]; then
    echo "停止 $c..."
    docker stop "$c" 2>/dev/null || true
    docker rm "$c" 2>/dev/null || true
    echo "✅ $c 已停止"
  fi
done
