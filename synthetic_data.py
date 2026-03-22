"""Generate synthetic learner data for POC demonstration."""

import json
import os
import random
from datetime import datetime, timedelta

import config
from learner_profile import create_default_profile, save_profile


def generate_synthetic_profile(videos: list[dict], num_watched: int = 6) -> dict:
    """
    Generate a synthetic learner profile who has watched some videos,
    taken quizzes with varying performance, to simulate a real learner journey.
    """
    profile = create_default_profile("user_001")

    # Pick videos to "watch" (random selection)
    watched_videos = random.sample(videos, min(num_watched, len(videos)))

    base_time = datetime.now() - timedelta(days=num_watched + 1)

    for i, video in enumerate(watched_videos):
        watch_time = base_time + timedelta(days=i, hours=random.randint(9, 20))

        # Simulate varying engagement
        completion = random.uniform(0.6, 1.0)
        profile["watch_history"].append({
            "video_id": video["video_id"],
            "timestamp": watch_time.isoformat(),
            "completion_rate": round(completion, 2),
        })

        # Simulate quiz — performance improves slightly over time (learning effect)
        base_accuracy = 0.3 + (i * 0.05)  # gradual improvement
        noise = random.uniform(-0.1, 0.15)
        accuracy = min(0.95, max(0.1, base_accuracy + noise))

        num_questions = random.choice([4, 5, 5, 6])
        correct = round(accuracy * num_questions)
        actual_accuracy = correct / num_questions

        # Determine difficulty based on simulated learning stage
        if i < 2:
            difficulty = "easy"
        elif i < 4:
            difficulty = "medium"
        else:
            difficulty = random.choice(["medium", "hard"])

        # Generate synthetic questions
        questions = []
        concepts_for_video = _generate_concepts_for_topic(video["topic"])

        for j in range(num_questions):
            is_correct = j < correct
            concept = concepts_for_video[j % len(concepts_for_video)]
            questions.append({
                "question": f"Q{j+1} about {concept}",
                "concept": concept,
                "sub_topic": video["topic"],
                "difficulty": difficulty,
                "user_answer": "A" if is_correct else "B",
                "correct_answer": "A",
                "user_correct": is_correct,
            })
        random.shuffle(questions)  # shuffle so correct/wrong aren't grouped

        quiz_completion = random.uniform(0.7, 1.0)

        profile["quiz_history"].append({
            "video_id": video["video_id"],
            "topic": video["topic"],
            "timestamp": (watch_time + timedelta(minutes=10)).isoformat(),
            "num_questions": num_questions,
            "score": round(actual_accuracy, 3),
            "difficulty": difficulty,
            "completion_rate": round(quiz_completion, 2),
            "questions": questions,
        })

        # Update mastery scores
        alpha = 0.4
        current = profile["mastery_scores"].get(video["topic"], config.DEFAULT_MASTERY)
        profile["mastery_scores"][video["topic"]] = round(
            alpha * actual_accuracy + (1 - alpha) * current, 3
        )

        # Update concept mastery
        for q in questions:
            concept = q["concept"]
            current_cm = profile["concept_mastery"].get(concept, config.DEFAULT_MASTERY)
            q_score = 1.0 if q["user_correct"] else 0.0
            profile["concept_mastery"][concept] = round(
                alpha * q_score + (1 - alpha) * current_cm, 3
            )

    # Add some recall history (for older videos)
    if num_watched >= 3:
        for i in range(min(3, num_watched - 1)):
            video = watched_videos[i]
            concept = list(profile["concept_mastery"].keys())[i] if profile["concept_mastery"] else video["topic"]
            recall_correct = random.random() > 0.4  # 60% recall rate
            profile["recall_log"].append({
                "video_id": video["video_id"],
                "concept": concept,
                "question": f"Recall question about {concept}",
                "answer": "Learner's answer",
                "score": round(random.uniform(0.6, 0.9), 2) if recall_correct else round(random.uniform(0.1, 0.4), 2),
                "correct": recall_correct,
                "timestamp": (base_time + timedelta(days=i + 1, hours=9)).isoformat(),
            })

    # Schedule some pending recalls
    if num_watched >= 2:
        recent_video = watched_videos[-1]
        weakest = min(profile["concept_mastery"].items(), key=lambda x: x[1])
        profile["recall_queue"].append({
            "video_id": recent_video["video_id"],
            "concept": weakest[0],
            "scheduled_date": datetime.now().strftime("%Y-%m-%d"),
        })

    save_profile(profile)
    return profile


def _generate_concepts_for_topic(topic: str) -> list[str]:
    """Generate plausible concept names for a topic."""
    concept_map = {
        # Finance cluster
        "stock market": ["shares", "market capitalization", "dividends", "bull vs bear"],
        "stock exchange": ["trading floor", "bid-ask spread", "IPO process", "market makers"],
        "supply and demand": ["equilibrium price", "demand curve", "supply shift", "price elasticity"],
        "compound interest": ["principal amount", "interest rate", "compounding frequency", "time value"],
        "recession": ["GDP decline", "unemployment rise", "consumer spending", "business cycle"],
        "inflation": ["price levels", "purchasing power", "monetary policy", "CPI index"],
        "bonds": ["face value", "coupon rate", "yield", "maturity date"],
        # Biology / Science cluster
        "cell structure": ["cell membrane", "nucleus function", "mitochondria", "organelles"],
        "dna replication": ["helicase", "DNA polymerase", "leading strand", "Okazaki fragments"],
        "atomic structure": ["protons neutrons", "electron shells", "atomic number", "isotopes"],
        "natural selection": ["adaptation", "survival fitness", "genetic variation", "speciation"],
        # CS / Math
        "algorithms": ["step-by-step procedure", "efficiency", "sorting searching", "computational thinking"],
        "mutual funds": ["diversification", "NAV", "expense ratio", "fund manager"],
        "internet": ["TCP/IP protocol", "packets routing", "DNS lookup", "HTTP requests"],
        "biology overview": ["living systems", "cell theory", "evolution", "ecology basics"],
    }
    return concept_map.get(topic, [f"{topic} concept 1", f"{topic} concept 2",
                                    f"{topic} concept 3", f"{topic} concept 4"])


if __name__ == "__main__":
    # Load videos and generate profile
    from data_ingestion import load_videos
    videos = load_videos()
    profile = generate_synthetic_profile(videos, num_watched=6)
    print(f"Generated synthetic profile with {len(profile['watch_history'])} watched videos")
    print(f"Mastery scores: {json.dumps(profile['mastery_scores'], indent=2)}")
    print(f"Weak concepts: {[(c, s) for c, s in profile['concept_mastery'].items() if s < 0.5]}")
