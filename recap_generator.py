"""Personalized recap generation using RAG over learner's watch history."""

import json
from sentence_transformers import SentenceTransformer

import config
from data_ingestion import get_collection
from learner_profile import get_watched_video_ids, get_weak_concepts
from llm import generate_json


_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embed_model


def retrieve_relevant_past_context(video_transcript: str, profile: dict, top_k: int = None,
                                   current_video_id: str = None) -> list[dict]:
    """
    Dual-query retrieval from user's watch history:
      Query 1 — current video transcript (finds related past content)
      Query 2 — learner's weak concepts (finds content targeting weak areas)
    Results are merged and deduplicated by chunk id.
    Excludes the current video so we only surface genuinely past content.
    """
    top_k = top_k or config.TOP_K_RETRIEVAL
    collection = get_collection()
    model = get_embed_model()

    watched_ids = get_watched_video_ids(profile)
    # Exclude the current video — it was just watched, not "past" context
    if current_video_id:
        watched_ids = [vid for vid in watched_ids if vid != current_video_id]
    if not watched_ids:
        return []

    where_filter = {"video_id": {"$in": watched_ids}}

    # Query 1: current video content
    q1_embedding = model.encode(video_transcript).tolist()
    results_video = collection.query(
        query_embeddings=[q1_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    # Query 2: weak concepts
    weak = get_weak_concepts(profile)
    weak_text = " ".join(w["concept"] for w in weak[:8])
    results_weak = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    if weak_text.strip():
        q2_embedding = model.encode(weak_text).tolist()
        results_weak = collection.query(
            query_embeddings=[q2_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

    # Merge and deduplicate
    seen = set()
    contexts = []

    for source_results, label in [(results_video, "video"), (results_weak, "weak")]:
        if not source_results["documents"] or not source_results["documents"][0]:
            continue
        for doc, meta, dist in zip(
            source_results["documents"][0],
            source_results["metadatas"][0],
            source_results["distances"][0],
        ):
            chunk_key = f"{meta['video_id']}_{meta.get('chunk_index', 0)}"
            if chunk_key in seen:
                continue
            seen.add(chunk_key)
            contexts.append({
                "text": doc,
                "video_id": meta["video_id"],
                "title": meta["title"],
                "topic": meta["topic"],
                "similarity": 1 - dist,
                "retrieval_source": label,
            })

    # Sort by similarity descending, take top_k total
    contexts.sort(key=lambda x: x["similarity"], reverse=True)
    return contexts[:top_k]


def generate_recap(video: dict, profile: dict) -> dict:
    """
    Generate a personalized 3-bullet recap for a video.
    Returns: {bullets: [str, str, str], reasoning: str}
    """
    past_contexts = retrieve_relevant_past_context(video["transcript"], profile,
                                                    current_video_id=video["video_id"])
    weak = get_weak_concepts(profile)
    weak_list = [w["concept"] for w in weak[:5]]
    mastery = profile.get("mastery_scores", {})

    past_context_str = ""
    if past_contexts:
        for ctx in past_contexts:
            past_context_str += f"\n--- From '{ctx['title']}' (topic: {ctx['topic']}, relevance: {ctx['similarity']:.2f}) ---\n{ctx['text']}\n"
    else:
        past_context_str = "No relevant past videos found."

    prompt = f"""You are a learning coach creating a personalized 60-second recap for a learner.

CURRENT VIDEO:
Title: {video['title']}
Topic: {video['topic']}
Transcript: {video['transcript']}

LEARNER CONTEXT:
- Topics mastered (score > 0.7): {json.dumps({k: v for k, v in mastery.items() if v > 0.7})}
- Weak concepts: {json.dumps(weak_list)}
- Number of videos watched: {len(profile['watch_history'])}

RELEVANT PAST CONTENT (from videos the learner has watched):
{past_context_str}

RULES:
1. Generate EXACTLY 3 bullet points.
2. Each bullet must be <= 25 words.
3. Bullet 1: The single most important concept from this video.
4. Bullet 2: How this connects to something the learner previously studied (reference the specific past video/topic if possible). If no past context, connect to general prerequisite knowledge.
5. Bullet 3: Focus on a concept the learner is weakest on (from weak concepts list) that appears in this video. If no weak concepts match, highlight the most nuanced/tricky point.
6. Do NOT repeat concepts the learner has already mastered.
7. Use simple, clear language.

Respond in this exact JSON format:
{{
    "bullet_1": "...",
    "bullet_2": "...",
    "bullet_3": "...",
    "reasoning": "Brief explanation of personalization choices made"
}}"""

    try:
        result = generate_json(prompt, max_tokens=500)
    except Exception:
        result = {
            "bullet_1": "Key concept from the video.",
            "bullet_2": "Connection to your past learning.",
            "bullet_3": "Area to focus on for improvement.",
            "reasoning": "Failed to parse LLM response.",
        }

    return {
        "bullets": [result["bullet_1"], result["bullet_2"], result["bullet_3"]],
        "reasoning": result.get("reasoning", ""),
        "past_contexts_used": [
            {"title": c["title"], "topic": c["topic"], "similarity": round(c["similarity"], 3)}
            for c in past_contexts
        ],
    }
