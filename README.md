# Memory Server

A user memory management API powered by **mem0** + **Qdrant** + **OpenAI** + **MongoDB**. Intelligently store, search, and manage user memories with semantic understanding.

## Features

- **User Profile Management** (MongoDB): Persistent user data, preferences, and statistics
- **Semantic Memory Storage** (Qdrant + mem0): Store facts with automatic embedding
- **Smart Search**: Find relevant memories using semantic similarity
- **Conversation Memory Extraction**: Automatically extract important information from conversations
- **Batch Operations**: Add multiple memories at once
- **Multi-User Support**: Each user gets isolated memory storage

## Architecture

```
memory_server/
├── app/
│   ├── api/
│   │   ├── mem0_routes.py      # Memory API endpoints
│   │   └── user_routes.py       # User API endpoints
│   ├── core/
│   │   └── config.py            # Configuration management
│   ├── database/
│   │   └── mongodb.py           # MongoDB connection
│   ├── models/
│   │   └── schemas.py           # Pydantic schemas
│   └── services/
│       ├── mem0_service.py      # mem0 business logic
│       └── user_service.py      # User management logic
├── main.py                       # Application entry point
├── docker-compose.yml            # Qdrant + MongoDB + API
└── requirements.txt              # Dependencies
```

## Data Storage

| Component | Purpose | Storage |
|-----------|---------|---------|
| **MongoDB** | User profiles, preferences, stats | `users` collection |
| **Qdrant** | Semantic memory vectors | `mem0_{user_id}` collections |

## Quick Start

### Using Docker Compose (Recommended)

```bash
# 1. Copy environment variables
cp .env.example .env

# 2. Edit .env with your settings
# Set OPENAI_API_KEY and OPENAI_API_BASE

# 3. Start all services (Qdrant + MongoDB + API)
docker-compose up -d

# 4. Access API
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Local Development

```bash
# 1. Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Start Qdrant and MongoDB
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
docker run -p 27017:27017 mongo:8.0

# 3. Configure environment
cp .env.example .env
# Edit .env with your settings

# 4. Run the API
python main.py
```

## API Endpoints

### User Management

#### Create User
```http
POST /api/v1/users
Content-Type: application/json

{
  "user_id": "user123",
  "name": "Alice Chen",
  "email": "alice@example.com",
  "role": "software engineer",
  "preferences": {
    "language": "Python",
    "timezone": "UTC-8"
  }
}
```

#### Get User
```http
GET /api/v1/users/user123
```

#### Update User
```http
PUT /api/v1/users/user123
Content-Type: application/json

{
  "name": "Alice Chen",
  "preferences": {
    "language": "Python",
    "timezone": "UTC-8",
    "theme": "dark"
  }
}
```

#### List Users
```http
GET /api/v1/users?limit=50
```

#### Delete User
```http
DELETE /api/v1/users/user123
```

#### User Statistics
```http
GET /api/v1/users/user123/stats
```

### Memory Operations

#### Add Memory
```http
POST /api/v1/memories
Content-Type: application/json

{
  "user_id": "user123",
  "content": "I prefer working with Python over JavaScript",
  "metadata": {"category": "preference"}
}
```

**Note**: User info is automatically fetched from MongoDB for personalized memory extraction.

#### Search Memories
```http
POST /api/v1/memories/search
Content-Type: application/json

{
  "user_id": "user123",
  "query": "What programming languages do I like?",
  "limit": 5
}
```

#### Get All Memories
```http
GET /api/v1/memories/user123?limit=50
```

#### Update Memory
```http
PUT /api/v1/memories
Content-Type: application/json

{
  "user_id": "user123",
  "memory_id": "abc123",
  "content": "I prefer Python and TypeScript"
}
```

#### Delete Memory
```http
DELETE /api/v1/memories
Content-Type: application/json

{
  "user_id": "user123",
  "memory_id": "abc123"
}
```

### Batch Operations

#### Add Multiple Memories
```http
POST /api/v1/memories/batch
Content-Type: application/json

{
  "user_id": "user123",
  "memories": [
    "I love coffee in the morning",
    "I work best in the afternoon",
    "I prefer asynchronous communication"
  ]
}
```

### Conversation Memory

#### Extract Memories from Conversation
```http
POST /api/v1/memories/conversation
Content-Type: application/json

{
  "user_id": "user123",
  "messages": [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Hello Alice!"},
    {"role": "user", "content": "I'm a software engineer"}
  ]
}
```

### User Context (Memory-aware)

#### Set User Context
```http
POST /api/v1/memories/context
Content-Type: application/json

{
  "user_id": "user123",
  "name": "Alice Chen",
  "email": "alice@example.com",
  "role": "software engineer",
  "preferences": {
    "language": "Python",
    "timezone": "UTC-8"
  }
}
```

**Note**: This updates both MongoDB (persistent) and adds a context memory for mem0.

#### Get User Context
```http
GET /api/v1/memories/context/user123
```

## Configuration

Environment variables (`.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | **Required** |
| `OPENAI_API_BASE` | OpenAI API base URL | **Required** |
| `OPENAI_MODEL` | LLM model | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `QDRANT_HOST` | Qdrant host | `localhost` |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `MONGO_URI` | MongoDB URI | `mongodb://localhost:27017` |
| `MONGO_DB` | MongoDB database | `memory_server` |
| `VECTOR_SIZE` | Embedding dimensions | `1536` |

## How It Works

### User Data Flow

1. **User Creation**: Stored in MongoDB with profile, preferences, stats
2. **Memory Addition**: User info fetched from MongoDB → sent to mem0 for personalized extraction
3. **Auto-updates**: Memory count and last_active automatically updated in MongoDB

### Memory Storage

- Each user gets isolated Qdrant collection (`mem0_{user_id}`)
- Content embedded using OpenAI's `text-embedding-3-small`
- mem0 intelligently extracts facts using user context from MongoDB

### Example Flow

```python
# 1. Create user (stored in MongoDB)
POST /api/v1/users {"user_id": "alice", "name": "Alice"}

# 2. Add memory (user info auto-fetched from MongoDB)
POST /api/v1/memories {
  "user_id": "alice",
  "content": "I love Python"
}

# 3. Memory count auto-incremented in MongoDB
GET /api/v1/users/alice/stats
# Returns: {"memory_count": 1, "last_active": "..."}
```

## Example Usage

### Python Client

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Create user
requests.post(f"{BASE_URL}/users", json={
    "user_id": "alice",
    "name": "Alice Chen",
    "email": "alice@example.com"
})

# Add memory (user info auto-loaded from MongoDB)
requests.post(f"{BASE_URL}/memories", json={
    "user_id": "alice",
    "content": "I prefer Python over JavaScript"
})

# Search memories
response = requests.post(f"{BASE_URL}/memories/search", json={
    "user_id": "alice",
    "query": "programming preferences"
})
print(response.json())

# Get user stats
response = requests.get(f"{BASE_URL}/users/alice/stats")
print(response.json())
```

## Production Considerations

- Set `DEBUG=False` in production
- Use managed Qdrant and MongoDB instances for scalability
- Implement proper authentication/authorization
- Configure CORS appropriately for your domain
- Set up monitoring and logging
- Use a secrets manager for API keys

## License

MIT
