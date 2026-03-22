"""Next-day recall check: spaced repetition question generation and evaluation."""

import json

import config
from learner_profile import (
    load_profile, save_profile, get_due_recalls,
    schedule_recall, record_recall
)
from data_ingestion import load_videos
from llm import generate_json, generate


def find_weakest_concept_for_video(profile: dict, video_id: str) -> str:
    """Find the concept with lowest mastery from a specific video's quiz."""
    video_quizzes = [q for q in profile["quiz_history"] if q["video_id"] == video_id]
    if not video_quizzes:
        return None

    worst_concept = None
    worst_score = 1.0

    for quiz in video_quizzes:
        for question in quiz.get("questions", []):
            concept = question.get("concept", "")
            if concept and not question.get("user_correct", True):
                mastery = profile["concept_mastery"].get(concept, config.DEFAULT_MASTERY)
                if mastery < worst_score:
                    worst_score = mastery
                    worst_concept = concept

    if not worst_concept and video_quizzes:
        worst_concept = video_quizzes[-1].get("topic", "the topic")

    return worst_concept


def generate_recall_question(video: dict, concept: str, profile: dict) -> dict:
    """
    Generate a recall question (fill-in-the-blank or free-recall, NOT MCQ).
    Tests actual recall, not recognition.
    """
    past_recall = [r for r in profile["recall_log"]
                   if r["video_id"] == video["video_id"] and r["concept"] == concept]

    avoid = ""
    if past_recall:
        avoid = f"\nAvoid similar questions to: {past_recall[-1]['question']}"

    prompt = f"""Generate ONE recall question to test if a learner remembers a concept from a video they watched.

VIDEO: "{video['title']}" (topic: {video['topic']})
CONCEPT TO TEST: {concept}
TRANSCRIPT: {video['transcript']}
{avoid}

RULES:
1. Use FILL-IN-THE-BLANK or SHORT ANSWER format (NOT multiple choice).
2. The question should test RECALL (pulling from memory), not RECOGNITION.
3. Keep it concise - answerable in 1-2 sentences.
4. Include enough context that the learner knows what video/topic you're referring to.

Return JSON:
{{
    "question": "the recall question",
    "expected_answer": "the ideal answer (1-2 sentences)",
    "concept": "{concept}",
    "question_type": "fill_blank" or "short_answer"
}}"""

    try:
        result = generate_json(prompt, max_tokens=300)
    except Exception:
        result = {
            "question": f"In your own words, explain what '{concept}' means in the context of {video['topic']}.",
            "expected_answer": "",
            "concept": concept,
            "question_type": "short_answer",
        }

    result["video_id"] = video["video_id"]
    result["video_title"] = video["title"]
    return result


def evaluate_recall_answer(question: dict, user_answer: str) -> dict:
    """
    Evaluate a free-text recall answer using structured rubric scoring.
    Scores 3 dimensions in a single LLM call for consistency.
    Returns score (0-1) and feedback.
    """
    prompt = f"""You are evaluating a learner's recall of a concept using a structured rubric.

QUESTION: {question['question']}
EXPECTED ANSWER: {question['expected_answer']}
LEARNER'S ANSWER: {user_answer}

Score each dimension from 0.0 to 1.0:

1. **factual_accuracy**: Are the facts stated correct?
   - 1.0 = all facts correct, 0.5 = mix of correct/incorrect, 0.0 = all wrong or "I don't know"

2. **completeness**: Does the answer cover the key points?
   - 1.0 = all key points covered, 0.5 = partial, 0.0 = missing everything important

3. **key_term_usage**: Does the learner use the correct terminology?
   - 1.0 = uses proper terms, 0.5 = vague/informal language, 0.0 = no relevant terms

Return JSON:
{{
    "factual_accuracy": <float 0-1>,
    "completeness": <float 0-1>,
    "key_term_usage": <float 0-1>,
    "score": <average of the three dimensions>,
    "feedback": "brief feedback for the learner (1 sentence)",
    "correct_answer_summary": "the key point they should remember"
}}"""

    try:
        result = generate_json(prompt, max_tokens=300, model="evaluation")
        # Recompute score as average of 3 dimensions for consistency
        dims = [result.get("factual_accuracy", 0.5),
                result.get("completeness", 0.5),
                result.get("key_term_usage", 0.5)]
        result["score"] = round(sum(dims) / len(dims), 3)
    except Exception:
        result = {"score": 0.5, "factual_accuracy": 0.5, "completeness": 0.5,
                  "key_term_usage": 0.5, "feedback": "Could not evaluate.",
                  "correct_answer_summary": ""}

    return result


def get_todays_recall_questions(profile: dict) -> list[dict]:
    """Get all recall questions due today."""
    due = get_due_recalls(profile)
    if not due:
        return []

    videos = load_videos()
    video_map = {v["video_id"]: v for v in videos}

    questions = []
    for recall_item in due:
        video = video_map.get(recall_item["video_id"])
        if not video:
            continue
        question = generate_recall_question(video, recall_item["concept"], profile)
        questions.append(question)

    return questions


def schedule_recalls_for_video(profile: dict, video_id: str, quiz_result: dict = None) -> dict:
    """
    After watching a video and taking a quiz, schedule recall checks.
    Picks the weakest concept for D+1 recall.
    """
    concept = find_weakest_concept_for_video(profile, video_id)
    if not concept:
        videos = load_videos()
        video_map = {v["video_id"]: v for v in videos}
        video = video_map.get(video_id)
        concept = video["topic"] if video else "the topic"

    profile = schedule_recall(profile, video_id, concept, days_from_now=1)

    return {
        "scheduled": True,
        "video_id": video_id,
        "concept": concept,
        "scheduled_for": "tomorrow",
        "message": f"Recall check scheduled for tomorrow on '{concept}'.",
    }
