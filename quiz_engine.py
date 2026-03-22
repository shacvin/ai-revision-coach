"""Adaptive quiz generation with real-time difficulty adjustment."""

import json

import config
from learner_profile import calculate_difficulty, calculate_quiz_length
from llm import generate_json


def extract_concepts(video: dict) -> list[dict]:
    """Extract key concepts from a video transcript using LLM."""
    prompt = f"""Extract the key concepts taught in this educational video.

Title: {video['title']}
Topic: {video['topic']}
Transcript: {video['transcript']}

Return a JSON array of concepts, each with:
- "concept": short name of the concept (2-5 words)
- "sub_topic": the broader sub-topic it belongs to
- "description": one sentence explaining the concept
- "importance": "high", "medium", or "low"

Return 5-8 concepts, ordered by importance. Respond with ONLY the JSON array."""

    try:
        concepts = generate_json(prompt, max_tokens=800)
        if isinstance(concepts, dict):
            concepts = [concepts]
    except Exception:
        concepts = [{"concept": video["topic"], "sub_topic": video["topic"],
                     "description": "Main topic", "importance": "high"}]
    return concepts


def generate_questions(video: dict, concepts: list[dict], difficulty: str,
                       num_questions: int, profile: dict) -> list[dict]:
    """Generate quiz questions at the specified difficulty level."""
    weak_concepts = [c for c, m in profile.get("concept_mastery", {}).items() if m < 0.5]

    difficulty_guidelines = {
        "easy": "Simple recall and definition questions. Test if the learner remembers basic facts.",
        "medium": "Application questions. Test if the learner can apply concepts to new scenarios.",
        "hard": "Analysis and comparison questions. Test deeper understanding, edge cases, and connections between concepts.",
    }

    prompt = f"""Generate {num_questions} multiple-choice quiz questions for this educational video.

VIDEO:
Title: {video['title']}
Topic: {video['topic']}
Transcript: {video['transcript']}

KEY CONCEPTS:
{json.dumps(concepts, indent=2)}

DIFFICULTY: {difficulty.upper()}
Guidelines: {difficulty_guidelines[difficulty]}

LEARNER WEAK CONCEPTS (give extra attention to these if they appear):
{json.dumps(weak_concepts[:5])}

RULES:
1. Each question must have exactly 4 options (A, B, C, D).
2. Only ONE correct answer per question.
3. Wrong answers should be plausible but clearly incorrect to someone who understood the content.
4. Cover different concepts - don't repeat the same concept.
5. If a weak concept is relevant to this video, include at least one question on it.
6. Keep questions under 2 sentences. Keep each option under 15 words. Keep explanations under 20 words.

Return a JSON array where each element has:
- "question": the question text (concise)
- "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}
- "correct_answer": "A", "B", "C", or "D"
- "concept": which concept this tests
- "sub_topic": broader sub-topic
- "difficulty": "{difficulty}"
- "explanation": brief explanation of the correct answer (1 sentence)

Respond with ONLY the JSON array."""

    try:
        questions = generate_json(prompt, max_tokens=5000)
        if isinstance(questions, dict):
            questions = [questions]
    except Exception as e:
        print(f"  [quiz_engine] generate_questions failed: {e}")
        questions = []

    for q in questions:
        q["difficulty"] = difficulty
        q["user_answer"] = None
        q["user_correct"] = None

    return questions[:num_questions]


def adapt_difficulty(questions_so_far: list[dict], current_difficulty: str) -> str:
    """
    Adapt difficulty based on real-time performance.
    Called after every ADAPTATION_INTERVAL questions.
    """
    answered = [q for q in questions_so_far if q.get("user_answer") is not None]
    if not answered:
        return current_difficulty

    correct = sum(1 for q in answered if q.get("user_correct"))
    accuracy = correct / len(answered)

    difficulty_order = ["easy", "medium", "hard"]
    current_idx = difficulty_order.index(current_difficulty)

    if accuracy > 0.8 and current_idx < 2:
        return difficulty_order[current_idx + 1]
    elif accuracy < 0.4 and current_idx > 0:
        return difficulty_order[current_idx - 1]
    return current_difficulty


def generate_adaptive_quiz(video: dict, profile: dict) -> dict:
    """
    Generate a full adaptive quiz for a video.
    Returns the initial batch of questions; adaptation happens during answering.
    """
    concepts = extract_concepts(video)

    difficulty, difficulty_score = calculate_difficulty(
        profile, video["topic"],
        related_topics=[c["sub_topic"] for c in concepts]
    )
    quiz_length = calculate_quiz_length(profile)

    questions = generate_questions(video, concepts, difficulty, quiz_length, profile)

    return {
        "video_id": video["video_id"],
        "video_title": video["title"],
        "topic": video["topic"],
        "concepts": concepts,
        "initial_difficulty": difficulty,
        "difficulty_score": round(difficulty_score, 3),
        "quiz_length": quiz_length,
        "questions": questions,
        "current_difficulty": difficulty,
    }


def generate_additional_questions(video: dict, concepts: list[dict],
                                  difficulty: str, num: int, profile: dict,
                                  missed_concepts: list[str] = None) -> list[dict]:
    """Generate additional questions, optionally focusing on missed concepts."""
    focus = ""
    if missed_concepts:
        focus = f"\nFOCUS on these concepts the learner got wrong: {json.dumps(missed_concepts)}"

    difficulty_guidelines = {
        "easy": "Simple recall and definition questions.",
        "medium": "Application questions requiring concept use.",
        "hard": "Analysis, comparison, and edge case questions.",
    }

    prompt = f"""Generate {num} additional multiple-choice quiz questions.

VIDEO:
Title: {video['title']}
Topic: {video['topic']}
Transcript: {video['transcript']}

CONCEPTS: {json.dumps([c['concept'] for c in concepts])}
DIFFICULTY: {difficulty.upper()} - {difficulty_guidelines[difficulty]}
{focus}

Return a JSON array. Each element:
- "question", "options" (A/B/C/D), "correct_answer", "concept", "sub_topic", "difficulty", "explanation"

ONLY the JSON array."""

    try:
        questions = generate_json(prompt, max_tokens=1200)
        if isinstance(questions, dict):
            questions = [questions]
    except Exception:
        questions = []

    for q in questions:
        q["difficulty"] = difficulty
        q["user_answer"] = None
        q["user_correct"] = None
        q["is_adaptive"] = True

    return questions[:num]


def process_answer(quiz: dict, question_idx: int, user_answer: str) -> dict:
    """
    Process a user's answer and potentially trigger adaptation.
    Returns updated quiz state with adaptation info.
    """
    q = quiz["questions"][question_idx]
    q["user_answer"] = user_answer
    q["user_correct"] = user_answer == q["correct_answer"]

    answered_count = sum(1 for q in quiz["questions"] if q.get("user_answer") is not None)

    adaptation_info = {"adapted": False}

    if answered_count > 0 and answered_count % config.ADAPTATION_INTERVAL == 0:
        new_difficulty = adapt_difficulty(quiz["questions"], quiz["current_difficulty"])
        if new_difficulty != quiz["current_difficulty"]:
            adaptation_info = {
                "adapted": True,
                "old_difficulty": quiz["current_difficulty"],
                "new_difficulty": new_difficulty,
                "reason": f"After {answered_count} questions, accuracy triggered difficulty change",
            }
            quiz["current_difficulty"] = new_difficulty

    return {
        "correct": q["user_correct"],
        "correct_answer": q["correct_answer"],
        "explanation": q.get("explanation", ""),
        "adaptation": adaptation_info,
        "questions_answered": answered_count,
        "questions_remaining": len(quiz["questions"]) - answered_count,
    }


def get_quiz_score(quiz: dict) -> dict:
    """Calculate final quiz score and per-concept breakdown."""
    answered = [q for q in quiz["questions"] if q.get("user_answer") is not None]
    if not answered:
        return {"score": 0, "total": 0, "accuracy": 0, "completion_rate": 0, "concept_scores": {}}

    correct = sum(1 for q in answered if q.get("user_correct"))
    total = len(answered)

    concept_scores = {}
    for q in answered:
        concept = q.get("concept", "unknown")
        if concept not in concept_scores:
            concept_scores[concept] = {"correct": 0, "total": 0}
        concept_scores[concept]["total"] += 1
        if q.get("user_correct"):
            concept_scores[concept]["correct"] += 1

    for c in concept_scores:
        concept_scores[c]["accuracy"] = (
            concept_scores[c]["correct"] / concept_scores[c]["total"]
        )

    return {
        "score": correct,
        "total": total,
        "accuracy": round(correct / total, 3),
        "completion_rate": round(total / len(quiz["questions"]), 3),
        "concept_scores": concept_scores,
        "difficulty_changes": quiz.get("current_difficulty") != quiz.get("initial_difficulty"),
        "final_difficulty": quiz["current_difficulty"],
    }
