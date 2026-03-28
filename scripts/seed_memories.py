#!/usr/bin/env python3
"""批量 POST /api/v1/memories 写入测试数据。

  MEMORY_SERVER_URL       默认 http://localhost:8000
  MEMORY_SERVER_TOKEN     必填，Bearer 里的 JWT
  MEMORY_SERVER_COUNT     默认 100
  MEMORY_SERVER_NAMESPACE 默认 main
  MEMORY_SERVER_SUBJECT_ID 默认 string

示例：

  export MEMORY_SERVER_TOKEN='你的JWT'
  python scripts/seed_memories.py
"""

from __future__ import annotations

import os
import sys

import httpx

BASE = os.environ.get("MEMORY_SERVER_URL", "http://localhost:8000").rstrip("/")
TOKEN = os.environ.get("MEMORY_SERVER_TOKEN", "")
COUNT = int(os.environ.get("MEMORY_SERVER_COUNT", "100"))
NAMESPACE = os.environ.get("MEMORY_SERVER_NAMESPACE", "main")
SUBJECT_ID = os.environ.get("MEMORY_SERVER_SUBJECT_ID", "string")

# 轮换语料，与序号拼成 100 条不重复正文
_FRAGMENTS = [
    "我喜欢踢足球",
    "周末常和朋友打篮球",
    "最近在学 Python 异步编程",
    "早餐习惯喝牛奶",
    "对海鲜过敏",
    "老家在成都",
    "通勤坐地铁二号线",
    "睡前会听播客",
    "不喜欢太甜的食物",
    "计划明年去日本旅行",
    "养了一只橘猫",
    "工作做后端开发",
    "常用编辑器是 VS Code",
    "咖啡只喝美式",
    "下雨喜欢宅家看电影",
    "会一点吉他",
    "目标是保持健康作息",
    "讨厌长时间开会",
    "周末喜欢徒步",
    "正在读一本科幻小说",
    "家里用双显示器",
    "习惯用深色主题",
    "对辣味接受度中等",
    "每年体检一次",
    "喜欢听独立音乐",
    "车技一般不开高速",
    "更习惯用文字沟通",
    "收藏了很多技术博客",
    "最近在戒含糖饮料",
    "希望多学一门外语",
]


def _content(i: int) -> str:
    base = _FRAGMENTS[i % len(_FRAGMENTS)]
    return f"[测试数据 {i + 1}/{COUNT}] {base}"


def main() -> None:
    if not TOKEN:
        print("请设置环境变量 MEMORY_SERVER_TOKEN", file=sys.stderr)
        sys.exit(1)

    url = f"{BASE}/api/v1/memories"
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    ok = 0
    fail = 0
    with httpx.Client(timeout=120.0) as client:
        for i in range(COUNT):
            body = {
                "namespace": NAMESPACE,
                "subject_id": SUBJECT_ID,
                "content": _content(i),
                "metadata": {"seed_index": i + 1, "batch": "seed_memories"},
                "infer": True,
            }
            r = client.post(url, json=body, headers=headers)
            if r.status_code == 201:
                ok += 1
            else:
                fail += 1
                print(f"FAIL {i + 1} HTTP {r.status_code}: {r.text[:500]}", file=sys.stderr)
    print(f"完成：成功 {ok}，失败 {fail}，合计 {COUNT}")


if __name__ == "__main__":
    main()
