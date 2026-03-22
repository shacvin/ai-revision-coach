"""Learner profile management: mastery tracking, difficulty calculation, weakness detection."""

import json
import os
from datetime import datetime, timedelta

import config


def create_default_profile(user_id: str = "user_001") -> dict:
    """Create a default learner profile."""
    return {
        "user_id": user_id,
        "watch_history": [],  # list of {video_id, timestamp, completion_rate}
        "quiz_history": [],   # list of {video_id, timestamp, questions, score, difficulty, completion_rate}
        "mastery_scores": {}, # topic -> float (0-1)
        "concept_mastery": {},# concept -> float (0-1)
        "recall_log": [],     # list of {video_id, concept, question, answer, correct, timestamp}
        "recall_queue": [],   # list of {video_id, concept, scheduled_date}
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


def load_profile() -> dict:
    """Load learner profile from file."""
    if os.path.exists(config.LEARNER_FILE):
        with open(config.LEARNER_FILE) as f:
            return json.load(f)
    profile = create_default_profile()
    save_profile(profile)
    return profile


def save_profile(profile: dict):
    """Save learner profile to file."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    profile["updated_at"] = datetime.now().isoformat()
    with open(config.LEARNER_FILE, "w") as f:
        json.dump(profile, f, indent=2)


def record_watch(profile: dict, video_id: str, completion_rate: float = 1.0) -> dict:
    """Record that the user watched a video."""
    profile["watch_history"].append({
        "video_id": video_id,
        "timestamp": datetime.now().isoformat(),
        "completion_rate": completion_rate,
    })
    save_profile(profile)
    return profile


def _adaptive_alpha(profile: dict, topic: str) -> float:
    """Compute adaptive EMA alpha: responsive early, stable with more data."""
    n = len([q for q in profile["quiz_history"] if q["topic"] == topic])
    return max(0.3, 0.6 - 0.05 * n)


def record_quiz_result(profile: dict, video_id: str, topic: str, questions: list[dict],
                       score: float, difficulty: str, completion_rate: float) -> dict:
    """Record quiz results and update mastery scores."""
    profile["quiz_history"].append({
        "video_id": video_id,
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "score": score,
        "difficulty": difficulty,
        "completion_rate": completion_rate,
        "questions": questions,
    })

    # Update topic mastery (adaptive exponential moving average)
    alpha = _adaptive_alpha(profile, topic)
    current_mastery = profile["mastery_scores"].get(topic, config.DEFAULT_MASTERY)
    profile["mastery_scores"][topic] = alpha * score + (1 - alpha) * current_mastery

    # Update concept-level mastery
    for q in questions:
        concept = q.get("concept", "")
        if concept:
            current = profile["concept_mastery"].get(concept, config.DEFAULT_MASTERY)
            q_correct = 1.0 if q.get("user_correct", False) else 0.0
            profile["concept_mastery"][concept] = alpha * q_correct + (1 - alpha) * current

    save_profile(profile)
    return profile


def record_recall(profile: dict, video_id: str, concept: str,
                  question: str, answer: str, score: float) -> dict:
    """Record a recall check result."""
    correct = score >= 0.5
    profile["recall_log"].append({
        "video_id": video_id,
        "concept": concept,
        "question": question,
        "answer": answer,
        "score": score,
        "correct": correct,
        "timestamp": datetime.now().isoformat(),
    })

    # Update concept mastery (slightly slower alpha for recall)
    alpha = 0.3
    current = profile["concept_mastery"].get(concept, config.DEFAULT_MASTERY)
    profile["concept_mastery"][concept] = alpha * score + (1 - alpha) * current

    # If wrong, re-queue for D+3
    if not correct:
        scheduled = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        profile["recall_queue"].append({
            "video_id": video_id,
            "concept": concept,
            "scheduled_date": scheduled,
        })

    # Remove from recall queue if it was there
    profile["recall_queue"] = [
        r for r in profile["recall_queue"]
        if not (r["video_id"] == video_id and r["concept"] == concept
                and r["scheduled_date"] <= datetime.now().strftime("%Y-%m-%d"))
    ]

    save_profile(profile)
    return profile


def schedule_recall(profile: dict, video_id: str, concept: str, days_from_now: int = 1) -> dict:
    """Schedule a recall check."""
    scheduled = (datetime.now() + timedelta(days=days_from_now)).strftime("%Y-%m-%d")
    # Avoid duplicates
    existing = [r for r in profile["recall_queue"]
                if r["video_id"] == video_id and r["concept"] == concept]
    if not existing:
        profile["recall_queue"].append({
            "video_id": video_id,
            "concept": concept,
            "scheduled_date": scheduled,
        })
    save_profile(profile)
    return profile


def get_due_recalls(profile: dict) -> list[dict]:
    """Get recall checks due today or earlier."""
    today = datetime.now().strftime("%Y-%m-%d")
    return [r for r in profile["recall_queue"] if r["scheduled_date"] <= today]


def get_watched_video_ids(profile: dict) -> list[str]:
    """Get list of all watched video IDs."""
    return list({w["video_id"] for w in profile["watch_history"]})


def calculate_difficulty(profile: dict, topic: str, related_topics: list[str] = None) -> tuple[str, float]:
    """
    Calculate appropriate difficulty level for a quiz.
    Returns (difficulty_label, difficulty_score).
    """
    # Brand new learner — always start easy
    if not profile["quiz_history"]:
        return "easy", 0.1

    # Topic quiz accuracy
    topic_quizzes = [q for q in profile["quiz_history"] if q["topic"] == topic]
    if topic_quizzes:
        topic_accuracy = sum(q["score"] for q in topic_quizzes) / len(topic_quizzes)
    else:
        topic_accuracy = config.DEFAULT_MASTERY

    # Related topic accuracy
    related_accuracy = config.DEFAULT_MASTERY
    if related_topics:
        related_quizzes = [q for q in profile["quiz_history"] if q["topic"] in related_topics]
        if related_quizzes:
            related_accuracy = sum(q["score"] for q in related_quizzes) / len(related_quizzes)

    # Average attempts (approximated from completion rate)
    if topic_quizzes:
        avg_completion = sum(q["completion_rate"] for q in topic_quizzes) / len(topic_quizzes)
        attempt_factor = 1 - avg_completion  # low completion = struggling
    else:
        attempt_factor = 1 - config.DEFAULT_MASTERY

    # Weighted score
    difficulty_score = (
        0.4 * (1 - topic_accuracy) +
        0.3 * (1 - related_accuracy) +
        0.3 * attempt_factor
    )

    if difficulty_score < config.EASY_THRESHOLD:
        return "easy", difficulty_score
    elif difficulty_score < config.MEDIUM_THRESHOLD:
        return "medium", difficulty_score
    else:
        return "hard", difficulty_score


def calculate_quiz_length(profile: dict) -> int:
    """Determine quiz length based on past completion rates."""
    if not profile["quiz_history"]:
        return config.BASE_QUIZ_LENGTH

    recent = profile["quiz_history"][-5:]  # last 5 quizzes
    avg_completion = sum(q["completion_rate"] for q in recent) / len(recent)

    if avg_completion < 0.5:
        return max(config.MIN_QUIZ_LENGTH, config.BASE_QUIZ_LENGTH - 2)
    elif avg_completion > 0.8:
        return min(config.MAX_QUIZ_LENGTH, config.BASE_QUIZ_LENGTH + 2)
    return config.BASE_QUIZ_LENGTH


def get_weak_concepts(profile: dict, threshold: float = 0.5) -> list[dict]:
    """Get concepts where the learner is weak."""
    weak = []
    for concept, mastery in profile["concept_mastery"].items():
        if mastery < threshold:
            weak.append({"concept": concept, "mastery": mastery})
    weak.sort(key=lambda x: x["mastery"])
    return weak


def get_weak_topics(profile: dict, threshold: float = 0.5) -> list[dict]:
    """Get topics where the learner is weak."""
    weak = []
    for topic, mastery in profile["mastery_scores"].items():
        if mastery < threshold:
            weak.append({"topic": topic, "mastery": mastery})
    weak.sort(key=lambda x: x["mastery"])
    return weak


def get_mastery_summary(profile: dict) -> dict:
    """Get a summary of the learner's current state."""
    total_watched = len(profile["watch_history"])
    total_quizzes = len(profile["quiz_history"])
    avg_score = (sum(q["score"] for q in profile["quiz_history"]) / total_quizzes
                 if total_quizzes else 0)
    avg_completion = (sum(q["completion_rate"] for q in profile["quiz_history"]) / total_quizzes
                      if total_quizzes else 0)
    recall_attempts = len(profile["recall_log"])
    recall_correct = sum(1 for r in profile["recall_log"] if r["correct"])

    return {
        "total_videos_watched": total_watched,
        "total_quizzes_taken": total_quizzes,
        "average_quiz_score": round(avg_score, 3),
        "average_quiz_completion": round(avg_completion, 3),
        "recall_attempts": recall_attempts,
        "recall_accuracy": round(recall_correct / recall_attempts, 3) if recall_attempts else 0,
        "mastery_scores": profile["mastery_scores"],
        "weak_topics": get_weak_topics(profile),
        "weak_concepts": get_weak_concepts(profile),
        "pending_recalls": len(get_due_recalls(profile)),
    }
