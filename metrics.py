"""Learning impact metrics tracking and visualization."""

import json
import os
from datetime import datetime

import config
from learner_profile import load_profile


def compute_metrics(profile: dict) -> dict:
    """Compute all learning impact metrics from the learner profile."""

    # 1. Quiz score improvement over time
    quiz_scores = []
    for i, q in enumerate(profile["quiz_history"]):
        quiz_scores.append({
            "quiz_number": i + 1,
            "topic": q["topic"],
            "score": q["score"],
            "difficulty": q["difficulty"],
            "completion_rate": q["completion_rate"],
            "num_questions": q["num_questions"],
            "timestamp": q["timestamp"],
        })

    # Per-topic score progression
    topic_progression = {}
    for q in profile["quiz_history"]:
        topic = q["topic"]
        if topic not in topic_progression:
            topic_progression[topic] = []
        topic_progression[topic].append({
            "score": q["score"],
            "difficulty": q["difficulty"],
            "timestamp": q["timestamp"],
        })

    # 2. Difficulty progression
    difficulty_map = {"easy": 1, "medium": 2, "hard": 3}
    difficulty_progression = [
        {
            "quiz_number": i + 1,
            "difficulty_label": q["difficulty"],
            "difficulty_numeric": difficulty_map.get(q["difficulty"], 2),
            "topic": q["topic"],
        }
        for i, q in enumerate(profile["quiz_history"])
    ]

    # 3. Recall success rate
    recall_log = profile.get("recall_log", [])
    recall_metrics = {
        "total_attempts": len(recall_log),
        "correct": sum(1 for r in recall_log if r.get("correct")),
        "accuracy": (sum(1 for r in recall_log if r.get("correct")) / len(recall_log)
                     if recall_log else 0),
        "by_attempt": [
            {
                "attempt": i + 1,
                "concept": r["concept"],
                "score": r["score"],
                "correct": r["correct"],
            }
            for i, r in enumerate(recall_log)
        ],
    }

    # 4. Completion rate trend
    completion_trend = [
        {
            "quiz_number": i + 1,
            "completion_rate": q["completion_rate"],
            "topic": q["topic"],
        }
        for i, q in enumerate(profile["quiz_history"])
    ]

    # 5. Mastery evolution (snapshot current state)
    mastery_current = profile.get("mastery_scores", {})
    concept_mastery = profile.get("concept_mastery", {})

    # 6. Recommendation effectiveness
    # Track: after a recommended video, did the weak concept score improve?
    rec_effectiveness = _compute_recommendation_effectiveness(profile)

    # 7. Overall summary
    total_quizzes = len(profile["quiz_history"])
    if total_quizzes >= 2:
        first_half = profile["quiz_history"][:total_quizzes // 2]
        second_half = profile["quiz_history"][total_quizzes // 2:]
        avg_first = sum(q["score"] for q in first_half) / len(first_half)
        avg_second = sum(q["score"] for q in second_half) / len(second_half)
        improvement = avg_second - avg_first
    else:
        avg_first = avg_second = improvement = 0

    summary = {
        "total_videos_watched": len(profile["watch_history"]),
        "total_quizzes": total_quizzes,
        "overall_avg_score": round(sum(q["score"] for q in profile["quiz_history"]) / total_quizzes, 3) if total_quizzes else 0,
        "first_half_avg": round(avg_first, 3),
        "second_half_avg": round(avg_second, 3),
        "score_improvement": round(improvement, 3),
        "recall_accuracy": round(recall_metrics["accuracy"], 3),
        "avg_completion_rate": round(sum(q["completion_rate"] for q in profile["quiz_history"]) / total_quizzes, 3) if total_quizzes else 0,
        "topics_mastered": sum(1 for v in mastery_current.values() if v >= 0.7),
        "topics_weak": sum(1 for v in mastery_current.values() if v < 0.5),
    }

    metrics = {
        "summary": summary,
        "quiz_scores": quiz_scores,
        "topic_progression": topic_progression,
        "difficulty_progression": difficulty_progression,
        "recall_metrics": recall_metrics,
        "completion_trend": completion_trend,
        "mastery_current": mastery_current,
        "concept_mastery": concept_mastery,
        "recommendation_effectiveness": rec_effectiveness,
        "computed_at": datetime.now().isoformat(),
    }

    # Save metrics
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _compute_recommendation_effectiveness(profile: dict) -> dict:
    """
    Check if quiz scores improved on topics after the learner was recommended
    and watched a video on that topic.
    """
    # Simple heuristic: for each topic with ≥2 quiz attempts,
    # check if the latest score > first score
    effectiveness = {}
    for topic, scores in _group_quiz_by_topic(profile).items():
        if len(scores) >= 2:
            first_score = scores[0]["score"]
            last_score = scores[-1]["score"]
            effectiveness[topic] = {
                "first_score": round(first_score, 3),
                "last_score": round(last_score, 3),
                "improved": last_score > first_score,
                "delta": round(last_score - first_score, 3),
            }
    return effectiveness


def _group_quiz_by_topic(profile: dict) -> dict:
    """Group quiz history by topic."""
    grouped = {}
    for q in profile["quiz_history"]:
        topic = q["topic"]
        if topic not in grouped:
            grouped[topic] = []
        grouped[topic].append(q)
    return grouped
