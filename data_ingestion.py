"""Data ingestion: embed video chunks and index in ChromaDB."""

import json

import config


def build_vector_index(videos: list[dict]):
    """Embed all video chunks and store in ChromaDB."""
    from sentence_transformers import SentenceTransformer
    import chromadb

    print("Loading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    # Delete existing collection if it exists
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for video in videos:
        for j, chunk in enumerate(video["chunks"]):
            chunk_id = f"{video['video_id']}_chunk_{j}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "video_id": video["video_id"],
                "title": video["title"],
                "topic": video["topic"],
                "chunk_index": j,
            })

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=32)

    # Add in batches (ChromaDB limit)
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=all_ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=all_chunks[i:end],
            metadatas=all_metadatas[i:end],
        )

    print(f"Indexed {len(all_chunks)} chunks in ChromaDB")
    return collection


def load_videos() -> list[dict]:
    """Load videos from saved JSON."""
    with open(config.VIDEOS_FILE) as f:
        return json.load(f)


def get_collection():
    """Get the ChromaDB collection."""
    import chromadb
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return client.get_collection(config.COLLECTION_NAME)
