# Centralized SQL strings for testing and maintenance.

SUBJECT_UPSERT_TOUCH = """
INSERT INTO subjects (namespace, subject_id, last_active, created_at, updated_at, preferences, custom_data, memory_count)
VALUES ($1, $2, $3, $3, $3, '{}', '{}', 0)
ON CONFLICT (namespace, subject_id) DO UPDATE SET
    last_active = EXCLUDED.last_active,
    updated_at = EXCLUDED.updated_at
"""

SUBJECT_UPSERT_CONTEXT = """
INSERT INTO subjects (
    namespace, subject_id, name, email, role, preferences, custom_data,
    updated_at, created_at, last_active, memory_count
)
VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $8, $8, 0)
ON CONFLICT (namespace, subject_id) DO UPDATE SET
    name = COALESCE(EXCLUDED.name, subjects.name),
    email = COALESCE(EXCLUDED.email, subjects.email),
    role = COALESCE(EXCLUDED.role, subjects.role),
    preferences = EXCLUDED.preferences,
    custom_data = EXCLUDED.custom_data,
    updated_at = EXCLUDED.updated_at
"""

SUBJECT_SELECT = """
SELECT namespace, subject_id, name, email, role, preferences, custom_data,
       memory_count, created_at, updated_at, last_active
FROM subjects WHERE namespace = $1 AND subject_id = $2
"""

SUBJECT_INCREMENT_MEMORY = """
INSERT INTO subjects (namespace, subject_id, last_active, updated_at, created_at, preferences, custom_data, memory_count)
VALUES ($1, $2, $3, $3, $3, '{}', '{}', $4)
ON CONFLICT (namespace, subject_id) DO UPDATE SET
    last_active = EXCLUDED.last_active,
    updated_at = EXCLUDED.updated_at,
    memory_count = subjects.memory_count + EXCLUDED.memory_count
"""

SERVICE_CLIENT_SELECT = """
SELECT client_id, client_secret_hash, namespaces, description, is_active, created_at, updated_at
FROM service_clients WHERE client_id = $1
"""

SERVICE_CLIENT_INSERT = """
INSERT INTO service_clients (client_id, client_secret_hash, namespaces, description, created_at, updated_at, is_active)
VALUES ($1, $2, $3, $4, $5, $5, true)
"""

SERVICE_CLIENT_LIST = """
SELECT client_id, namespaces, description, is_active, created_at, updated_at
FROM service_clients WHERE is_active = true ORDER BY client_id
"""

SERVICE_CLIENT_UPDATE = """
UPDATE service_clients SET updated_at = $2
WHERE client_id = $1
"""

SERVICE_CLIENT_DELETE = """
DELETE FROM service_clients WHERE client_id = $1
"""

SERVICE_TOKEN_INSERT = """
INSERT INTO service_tokens (token_hash, service_id, namespaces, expires_at, created_at, updated_at, is_active)
VALUES ($1, $2, $3, $4, $5, $5, true)
"""

SERVICE_TOKEN_SELECT = """
SELECT token_hash, service_id, namespaces, expires_at, is_active
FROM service_tokens WHERE token_hash = $1
"""
