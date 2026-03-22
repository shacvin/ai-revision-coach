"""Next best video recommendation based on weak concepts."""

import json
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

import config
from data_ingestion import get_collection, load_videos
from learner_profile import get_weak_concepts, get_weak_topics, get_watched_video_ids


_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embed_model


def _compute_freshness(profile: dict, topic: str) -> float:
    """Return 0.5 if topic was quizzed within FRESHNESS_RECENCY_DAYS, else 1.0."""
    topic_quizzes = [q for q in profile["quiz_history"] if q["topic"] == topic]
    if not topic_quizzes:
        return 1.0
    last_ts = topic_quizzes[-1]["timestamp"]
    try:
        last_date = datetime.fromisoformat(last_ts)
        days_ago = (datetime.now() - last_date).days
    except (ValueError, TypeError):
        return 1.0
    return 0.5 if days_ago <= config.FRESHNESS_RECENCY_DAYS else 1.0


def _embedding_topic_overlap(model, video_topic: str, weak_topics: list[dict]) -> float:
    """Compute max cosine similarity between a video's topic and weak topics via embeddings."""
    if not weak_topics:
        return 0.0
    weak_topic_strings = [w["topic"] for w in weak_topics]
    embeddings = model.encode([video_topic] + weak_topic_strings)
    video_emb = embeddings[0]
    weak_embs = embeddings[1:]
    # Cosine similarities
    sims = np.dot(weak_embs, video_emb) / (
        np.linalg.norm(weak_embs, axis=1) * np.linalg.norm(video_emb) + 1e-8
    )
    return float(np.max(sims))


def recommend_next_video(profile: dict, quiz_result: dict = None, top_n: int = 3) -> list[dict]:
    """
    Recommend next videos based on weak concepts and quiz performance.
    Uses separate per-concept queries for precise retrieval.
    """
    videos = load_videos()
    collection = get_collection()
    model = get_embed_model()

    watched_ids = set(get_watched_video_ids(profile))

    weak_concepts = get_weak_concepts(profile, threshold=0.6)
    weak_topics = get_weak_topics(profile, threshold=0.6)

    # Fallback: if no weak topics (learner did well), use all studied topics so
    # topic_overlap still reflects relevance to the learner's study area.
    topics_for_overlap = weak_topics if weak_topics else [
        {"topic": t, "mastery": m} for t, m in profile.get("mastery_scores", {}).items()
    ]

    quiz_weak = []
    if quiz_result and "concept_scores" in quiz_result:
        for concept, scores in quiz_result["concept_scores"].items():
            if scores["accuracy"] < 0.5:
                quiz_weak.append(concept)

    # Build individual query terms
    query_terms = []
    query_terms.extend([w["concept"] for w in weak_concepts[:5]])
    query_terms.extend([w["topic"] for w in weak_topics[:3]])
    query_terms.extend(quiz_weak[:3])

    if not query_terms:
        unwatched = [v for v in videos if v["video_id"] not in watched_ids]
        if unwatched:
            return [{
                "video_id": v["video_id"],
                "title": v["title"],
                "topic": v["topic"],
                "score": 0.5,
                "semantic_similarity": 0.0,
                "topic_overlap": 0.0,
                "freshness": 1.0,
                "weak_concepts_addressed": [],
                "reason": "Explore a new topic you haven't studied yet.",
            } for v in unwatched[:top_n]]
        return []

    # Separate query per weak item, aggregate max similarity per video
    video_scores = {}  # vid_id -> {max_similarity, ...}

    for term in query_terms:
        term_embedding = model.encode(term).tolist()
        results = collection.query(
            query_embeddings=[term_embedding],
            n_results=15,
            include=["metadatas", "distances"],
        )
        if not results["metadatas"] or not results["metadatas"][0]:
            continue
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            vid_id = meta["video_id"]
            if vid_id in watched_ids:
                continue
            similarity = 1 - dist
            if vid_id not in video_scores:
                video_scores[vid_id] = {
                    "video_id": vid_id,
                    "title": meta["title"],
                    "topic": meta["topic"],
                    "max_similarity": similarity,
                }
            else:
                video_scores[vid_id]["max_similarity"] = max(
                    video_scores[vid_id]["max_similarity"], similarity
                )

    weak_concept_set = set(w["concept"] for w in weak_concepts)

    recommendations = []
    for vid_id, info in video_scores.items():
        semantic = info["max_similarity"]
        topic_overlap = _embedding_topic_overlap(model, info["topic"], topics_for_overlap)
        freshness = _compute_freshness(profile, info["topic"])

        score = (
            config.REC_WEIGHT_SEMANTIC * semantic +
            config.REC_WEIGHT_TOPIC_OVERLAP * topic_overlap +
            config.REC_WEIGHT_FRESHNESS * freshness
        )

        recommendations.append({
            "video_id": vid_id,
            "title": info["title"],
            "topic": info["topic"],
            "score": round(score, 3),
            "semantic_similarity": round(semantic, 3),
            "topic_overlap": round(topic_overlap, 3),
            "freshness": round(freshness, 1),
            "weak_concepts_addressed": [
                c for c in weak_concept_set
                if c.lower() in info["title"].lower() or c.lower() in info["topic"].lower()
            ],
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    recommendations = recommendations[:top_n]

    for rec in recommendations:
        rec["reason"] = _generate_recommendation_reason(rec, weak_concepts, quiz_weak)

    return recommendations


def _generate_recommendation_reason(rec: dict, weak_concepts: list, quiz_weak: list) -> str:
    """Generate a human-readable reason for the recommendation."""
    reasons = []

    if rec.get("weak_concepts_addressed"):
        reasons.append(f"covers concepts you're weak on: {', '.join(rec['weak_concepts_addressed'])}")

    if rec["topic_overlap"] > 0.6:
        reasons.append(f"closely related to your weak topic area")

    if rec["semantic_similarity"] > 0.5:
        reasons.append("closely related to concepts you need to strengthen")

    if rec.get("freshness", 1.0) < 1.0:
        reasons.append("(recently studied topic — review opportunity)")

    if not reasons:
        reasons.append(f"explores {rec['topic']} which complements your learning path")

    return "Recommended because it " + "; ".join(reasons) + "."
