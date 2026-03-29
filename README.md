# Memory Server

面向公司内部团队的 Memory API 服务。在开源 **mem0** 之上做了一层稳定、易集成的 **FastAPI** 封装：向量语义检索走 **Qdrant**，词面（BM25）检索与主体画像、服务凭证等关系数据走 **PostgreSQL（ParadeDB 兼容镜像）**。调用方用 **namespace + subject_id** 隔离数据，用服务 JWT 做鉴权。

## 能力概览

| 能力 | 说明 |
|------|------|
| 服务鉴权 | `client_id` / `client_secret` 换 JWT；可选 **PostgreSQL 托管客户端**，未建库时回退 `.env` 里的 `SERVICE_CLIENTS_JSON` |
| 管理端 | `X-Admin-Token` 保护 `/api/v1/admin/*`，可创建/列出/更新/删除服务客户端、重置密钥 |
| 记忆隔离 | `namespace` + `subject_id`（+ 可选 `run_id` 会话维度） |
| 写入 | 单条、批量、会话抽取（infer 可关） |
| 检索 | `mode`: **`hybrid`**（向量+BM25 加权融合）、**`vector`**、**`bm25`**（纯 BM25） |
| 主体画像 | Subject Context 写入/读取，用于增强 mem0 写入时的上下文 |
| 统计 | 按主体的记忆统计接口 |
| 健康检查 | `/health` 探测 PostgreSQL、Qdrant、OpenAI 配置 |

## 技术栈

- **API**: FastAPI、Pydantic v2  
- **Memory / 向量**: mem0、Qdrant  
- **词面索引**: ParadeDB BM25（`memory_lexical` 表，中文 Lindera 分词）  
- **关系与配置库**: PostgreSQL（推荐 `paradedb/paradedb` 镜像；asyncpg）  
- **LLM / Embedding**: OpenAI 兼容 API（`OPENAI_BASE_URL` + Key）  
- **鉴权**: JWT（python-jose）、bcrypt 客户端密钥  

## 架构要点

1. **向量记忆**：mem0 将记忆写入 Qdrant（集合名等由 `QDRANT_*` 与 mem0 配置决定）。  
2. **词面记忆**：成功写入向量侧后，服务会把文本同步到 `memory_lexical`，供 **`mode=bm25`** 或 hybrid 中的 BM25 一路使用。**若 PostgreSQL 未连接，词面表不会更新，bm25/hybrid 的词面结果会缺失。**  
3. **检索分数**：响应中提供 **`vector_score`**、**`lexical_score`**（BM25 原始分）与 **`match_sources`**（`vector` / `bm25`）。混合排序由服务端内部加权完成，不对外返回融合分。  
4. **`run_id`**：请求里若传 **空字符串** `""`，服务会视为「未限定 run」，与库中 `run_id IS NULL` 的行一致；避免误写成 `run_id = ''` 导致零结果。  
5. **检索调参**：向量阈值、BM25 下限、混合融合过滤、hybrid 权重等通过环境变量 **`MEM0_SEARCH_*` / `MEM0_HYBRID_*`** 配置，**不在**搜索请求体中暴露。

## 快速启动

### 方式一：`start.sh`（本机 Python + Docker 依赖）

```bash
cp .env.example .env
# 填写 DATABASE_URL、OPENAI_API_KEY、OPENAI_BASE_URL、SECRET_KEY
# 若尚未用管理端建客户端，需配置 SERVICE_CLIENTS_JSON 用于换票
./start.sh
```

脚本会：创建/使用 `venv`、安装依赖、启动 **qdrant** 与 **paradedb** 容器，然后执行 `python main.py`（`DEBUG=True` 时带热重载）。

### 方式二：Docker Compose（含 API 容器）

```bash
cp .env.example .env
# 容器内已指向 compose 网络中的 qdrant / paradedb，可覆盖 .env 中的 DATABASE_URL、QDRANT_HOST 等
docker compose up --build
```

数据目录默认 `./data`（`qdrant`、`paradedb` 子目录）；可通过环境变量 **`DATA_DIR`** 指向外挂盘。

### 访问地址

| 用途 | URL |
|------|-----|
| API | `http://localhost:8899` |
| Swagger | `http://localhost:8899/docs` |
| ReDoc | `http://localhost:8899/redoc` |
| 健康检查 | `GET http://localhost:8899/health` |

## 环境变量（必填与常用）

**应用启动强依赖（pydantic 校验）**

- `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`SECRET_KEY`

**数据库（全功能建议始终配置）**

- `DATABASE_URL`：默认指向本机 `localhost:5433` 的 ParadeDB（与 `docker-compose` 端口映射一致）  
- `DATABASE_SSL`：本地一般为 `false`  

**Qdrant**

- `QDRANT_HOST`、`QDRANT_PORT`、`QDRANT_MEM0_COLLECTION`、`VECTOR_SIZE` 等  

**内部客户端（回退/冷启动）**

- `SERVICE_CLIENTS_JSON`：JSON 对象，键为 `client_id`，值为 `secret` + `namespaces` 数组。PostgreSQL 可用时，优先使用库表 `service_clients` 中的记录（可通过管理端维护）。  

**管理端**

- `ADMIN_API_TOKEN`：请求头 `X-Admin-Token` 须与之完全一致。  

**检索与服务端融合（可选）**

- `MEM0_SEARCH_VECTOR_MIN_SCORE`：向量检索 `threshold`  
- `MEM0_SEARCH_BM25_MIN_SCORE`：BM25 原始分下限（可选）  
- `MEM0_SEARCH_MIN_HYBRID_SCORE`：混合检索内部加权分下限（可选）  
- `MEM0_HYBRID_VECTOR_WEIGHT`、`MEM0_HYBRID_BM25_WEIGHT`：hybrid 权重（会自动归一化；和 ≤0 时回退 0.7/0.3）  

**代理**：若使用 SOCKS，请在环境中写 **`socks5://`**；程序会将错误的 `socks://` 前缀自动改为 `socks5://`。

完整示例见仓库根目录 **`.env.example`**。

## 鉴权

### 1. 换取服务 JWT

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "client_id": "svc-agent",
  "client_secret": "replace-me"
}
```

响应（模型 **`ServiceTokenResponse`**）包含：`access_token`、`service_id`、`namespaces`、`expires_in`（秒）。后续请求：

```http
Authorization: Bearer <access_token>
```

JWT 中的 `namespaces` 决定可访问的 `namespace`；包含 `"*"` 表示不限制（慎用）。

### 2. 管理端（维护服务客户端）

所有路由前缀 **`/api/v1/admin`**，需请求头：

```http
X-Admin-Token: <ADMIN_API_TOKEN>
```

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/admin/clients` | 创建客户端，响应体一次性返回 `client_secret` |
| GET | `/admin/clients` | 列出客户端 |
| PATCH | `/admin/clients/{client_id}` | 更新 namespaces、描述、启用状态 |
| POST | `/admin/clients/{client_id}/reset-secret` | 重置密钥 |
| DELETE | `/admin/clients/{client_id}` | 删除客户端 |

创建客户端需要 **PostgreSQL 已连接**；否则管理端写入会失败。

## 业务 API 摘要

前缀均为 **`/api/v1`**，除 `/auth/token` 与根路径外，记忆与主体接口需要 **Bearer JWT** 且 **`namespace` 在令牌授权范围内**。

### 记忆 `/memories`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/memories` | 单条写入 |
| POST | `/memories/search` | 语义/词面/混合搜索 |
| GET | `/memories/{namespace}/{subject_id}` | 列举记忆（Query：`limit`、`run_id`） |
| PUT | `/memories` | 更新内容 |
| DELETE | `/memories` | 删除 |
| POST | `/memories/batch` | 批量写入 |
| POST | `/memories/conversation` | 从多轮对话抽取记忆 |

**搜索请求体（`MemorySearch`）**

- 必填：`namespace`、`subject_id`、`query`  
- 可选：`limit`（默认 10）、**`mode`**（`hybrid` / `vector` / `bm25`）、`run_id`、`filters`（metadata JSON 包含匹配，与向量侧 filters 语义一致）  
- **兼容旧客户端**：请求体仍可使用字段名 **`fusion`**（等价于 `mode`）；取值 **`lexical`** 会自动视为 **`bm25`**。  

**`filters` 注意**：`{}` 不会附加 metadata 条件；不要传空字符串 `run_id` 表示「不限 run」——应省略 `run_id` 或传 `null`（空字符串服务端也会按「不限 run」处理）。

### 主体 `/subjects`

| 方法 | 路径 | 说明 |
|------|------|------|
| PUT | `/subjects/{subject_id}/context` | 写入/更新主体画像（Body 含 `namespace`） |
| GET | `/subjects/{subject_id}/context` | 读取画像（Query：`namespace`） |
| GET | `/subjects/{subject_id}/stats` | 主体记忆统计（Query：`namespace`） |

## 请求/响应与错误

- 多数记忆接口成功时返回 **`MemoryResponse`**（`success`、`message`、`data`）或列表封装 **`MemoryListResponse`**（`success`、`data`、`count`）。  
- HTTP 层错误由全局 `HTTPException` 处理，JSON 形如：`success: false`、`code`（如 `HTTP_403`）、`message`；`DEBUG=true` 时未捕获异常可能带 `detail`。  
- **`/health`**：`status` 为 `healthy` 或 `degraded`；任一下游异常时整体可能返回 **503**。

## 项目结构

```text
main.py                 # FastAPI 应用、生命周期、CORS、健康检查
app/
  api/                  # auth_routes, mem0_routes, user_routes, admin_routes
  core/                 # config.py, deps.py（JWT、namespace、Admin）
  database/             # postgres.py（连接池、迁移 SQL）、sql.py
  models/               # Pydantic schemas
  services/             # auth_service, mem0_service, user_service
docker-compose.yml      # qdrant、paradedb、api
start.sh                # 本地依赖容器 + venv + main.py
tests/                  # pytest
```

## 开发自检

```bash
./venv/bin/python -m pytest tests -q
```

---

若你来自旧文档：本项目**不再使用 MongoDB**；主体与服务客户端等均已落在 **PostgreSQL（ParadeDB）** 中，与 **Qdrant + mem0** 共同构成完整记忆链路。
