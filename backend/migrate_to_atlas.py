"""
Migrate local MongoDB database to MongoDB Atlas.
Reads connection details strictly from environment variables (.env).
Never prints or logs passwords or credentials.
"""
import os
import sys
import io
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

from dotenv import load_dotenv
import pymongo

# Load environment variables from backend/.env
backend_dir = Path(__file__).resolve().parent
env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

atlas_url = os.getenv("MONGODB_URL")
db_name = os.getenv("MONGODB_DB_NAME", "fitness_chatbot")
local_url = "mongodb://localhost:27017"

if not atlas_url or "localhost" in atlas_url:
    print("Error: MONGODB_URL in .env is not configured for Atlas.", file=sys.stderr)
    sys.exit(1)

print("Connecting to local MongoDB and MongoDB Atlas...")
local_client = pymongo.MongoClient(local_url, serverSelectionTimeoutMS=5000)
atlas_client = pymongo.MongoClient(atlas_url, serverSelectionTimeoutMS=10000)

local_db = local_client[db_name]
atlas_db = atlas_client[db_name]

# Ping both databases
try:
    local_client.admin.command('ping')
    print("✓ Local MongoDB connected successfully.")
except Exception as e:
    print(f"Error connecting to local MongoDB: {e}", file=sys.stderr)
    sys.exit(1)

try:
    atlas_client.admin.command('ping')
    print("✓ MongoDB Atlas connected successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB Atlas: {e}", file=sys.stderr)
    sys.exit(1)

local_collections = sorted(local_db.list_collection_names())
print(f"Detected {len(local_collections)} collections in local '{db_name}': {local_collections}")

results = []

for col_name in local_collections:
    local_col = local_db[col_name]
    atlas_col = atlas_db[col_name]
    
    local_doc_count = local_col.count_documents({})
    print(f"\nMigrating collection: '{col_name}' ({local_doc_count} docs)...")
    
    # Check if target already has docs
    atlas_doc_count_before = atlas_col.count_documents({})
    if atlas_doc_count_before > 0:
        print(f"  Target collection already has {atlas_doc_count_before} docs. Clearing target collection to prevent duplicates...")
        atlas_col.delete_many({})
    
    # Batch copy documents to preserve exact BSON and ObjectIDs
    BATCH_SIZE = 1000
    cursor = local_col.find({})
    batch = []
    inserted_total = 0
    
    for doc in cursor:
        batch.append(doc)
        if len(batch) >= BATCH_SIZE:
            atlas_col.insert_many(batch, ordered=True)
            inserted_total += len(batch)
            print(f"  Inserted {inserted_total}/{local_doc_count} docs...")
            batch = []
            
    if batch:
        atlas_col.insert_many(batch, ordered=True)
        inserted_total += len(batch)
        print(f"  Inserted {inserted_total}/{local_doc_count} docs.")

    # Replicate Indexes
    local_indexes = list(local_col.list_indexes())
    created_indexes = 0
    for idx in local_indexes:
        idx_name = idx.get("name")
        if idx_name == "_id_":
            continue  # default index
        keys = list(idx["key"].items())
        options = {k: v for k, v in idx.items() if k not in ("v", "key", "ns")}
        try:
            atlas_col.create_index(keys, **options)
            created_indexes += 1
        except Exception as idx_err:
            print(f"  Warning: Could not create index {idx_name} on {col_name}: {idx_err}")

    # Verify counts and indexes
    atlas_doc_count = atlas_col.count_documents({})
    atlas_indexes = list(atlas_col.list_indexes())
    
    match = (local_doc_count == atlas_doc_count)
    status = "MATCH" if match else "MISMATCH"
    
    results.append({
        "collection": col_name,
        "local_docs": local_doc_count,
        "atlas_docs": atlas_doc_count,
        "local_indexes": len(local_indexes),
        "atlas_indexes": len(atlas_indexes),
        "status": status
    })

print("\n" + "=" * 80)
print(f"{'COLLECTION':<25} {'LOCAL DOCS':<12} {'ATLAS DOCS':<12} {'LOCAL IDX':<10} {'ATLAS IDX':<10} {'STATUS'}")
print("-" * 80)
all_matched = True
for r in results:
    if r["status"] != "MATCH":
        all_matched = False
    print(f"{r['collection']:<25} {r['local_docs']:<12} {r['atlas_docs']:<12} {r['local_indexes']:<10} {r['atlas_indexes']:<10} {r['status']}")
print("=" * 80)

if all_matched:
    print("✓ All collection document counts and structures matched 100% between local and Atlas!")
else:
    print("❌ Discrepancies detected between local and Atlas!", file=sys.stderr)
    sys.exit(1)
