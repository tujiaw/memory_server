# Memory Server

面向公司内部团队的 Memory 平台 API。项目集成开源 `mem0`，通过 `FastAPI` 提供稳定、易用的服务到服务接口，帮助 AI 项目高效接入长期 memory，同时保留对效果的基本控制能力。

## 当前能力

- 服务到服务鉴权：内部服务使用 `client_id/client_secret` 换取 JWT
- `namespace + subject_id` 隔离：不同团队和业务主体分离存储
- 单条写入、批量写入、会话抽取、语义检索
- Subject Context：把主体画像存入 MongoDB，用于增强 memory 写入效果
- 健康检查：真实探测 MongoDB、Qdrant 和 OpenAI 配置状态

## 技术栈

- API: `FastAPI`
- Memory: `mem0`
- Vector Store: `Qdrant`
- Context Storage: `MongoDB`
- LLM/Embedding: `OpenAI`

## 快速启动

### 方式一：本地脚本

```bash
cp .env.example .env
# 编辑 .env，至少填写 OPENAI_API_KEY、SECRET_KEY、SERVICE_CLIENTS_JSON
./start.sh
```

### 方式二：Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

服务默认地址：

- API: `http://localhost:8899`
- Docs: `http://localhost:8899/docs`
- Health: `http://localhost:8899/health`

## 必填配置

`.env` 至少需要配置：

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `SECRET_KEY`
- `SERVICE_CLIENTS_JSON`

`SERVICE_CLIENTS_JSON` 示例：

```json
{
  "svc-agent": {
    "secret": "replace-me",
    "scopes": ["memory:read", "memory:write", "context:read", "context:write"],
    "namespaces": ["team-a"]
  }
}
```

## 鉴权流程

先换取访问令牌：

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "client_id": "svc-agent",
  "client_secret": "replace-me"
}
```

成功后在后续请求中带上：

```http
Authorization: Bearer <access_token>
```

## 核心接口

### 1. 写入单条 Memory

```http
POST /api/v1/memories
Authorization: Bearer <token>
Content-Type: application/json

{
  "namespace": "team-a",
  "subject_id": "user-123",
  "run_id": "session-001",
  "content": "The user prefers Python for backend work.",
  "metadata": {
    "category": "preference"
  }
}
```

### 2. 搜索 Memory

```http
POST /api/v1/memories/search
Authorization: Bearer <token>
Content-Type: application/json

{
  "namespace": "team-a",
  "subject_id": "user-123",
  "run_id": "session-001",
  "query": "What backend language does the user prefer?",
  "limit": 5,
  "filters": {
    "category": "preference"
  }
}
```

### 3. 批量写入

```http
POST /api/v1/memories/batch
Authorization: Bearer <token>
Content-Type: application/json

{
  "namespace": "team-a",
  "subject_id": "user-123",
  "memories": [
    { "content": "Works best in the morning." },
    { "content": "Prefers concise answers.", "metadata": { "category": "style" } }
  ]
}
```

### 4. 从会话中抽取 Memory

```http
POST /api/v1/memories/conversation
Authorization: Bearer <token>
Content-Type: application/json

{
  "namespace": "team-a",
  "subject_id": "user-123",
  "run_id": "chat-42",
  "messages": [
    { "role": "user", "content": "I usually write Python services." },
    { "role": "assistant", "content": "Got it." }
  ]
}
```

### 5. 写入 Subject Context

```http
PUT /api/v1/subjects/user-123/context
Authorization: Bearer <token>
Content-Type: application/json

{
  "namespace": "team-a",
  "name": "Alice",
  "role": "software engineer",
  "preferences": {
    "language": "Python",
    "response_style": "concise"
  }
}
```

## 返回与错误

成功响应统一为：

```json
{
  "success": true,
  "message": "optional",
  "data": {}
}
```

失败响应统一为：

```json
{
  "success": false,
  "code": "HTTP_403",
  "message": "Missing required scope: memory:write"
}
```

## 项目结构

```text
app/
  api/         # 路由层
  core/        # 配置与依赖注入
  database/    # MongoDB 连接
  models/      # Pydantic schemas
  services/    # auth / mem0 / subject context 业务逻辑
main.py        # 应用入口与全局错误/健康检查
tests/         # 最小回归测试
```

## 开发验证

```bash
./venv/bin/python -m pytest tests -q
```
