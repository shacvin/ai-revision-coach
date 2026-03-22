"""End-to-end backend tests for the AI Revision Coach.
Run each module in isolation, then test the full pipeline.

Tests cover: config, data loading, learner profile (adaptive alpha, default mastery),
LLM, recap (dual retrieval), quiz (full transcript), recommender (per-concept queries,
embedding topic overlap, freshness), recall (structured rubric), metrics, full pipeline.
"""

import json
import sys
import traceback

# ══════════════════════════════════════════════════════════════
# TEST 1: Config & Data Loading
# ══════════════════════════════════════════════════════════════
def test_config():
    print("=" * 60)
    print("TEST 1: Config & Environment")
    print("=" * 60)
    import config

    assert config.GOOGLE_API_KEY, "GOOGLE_API_KEY not set in .env"
    assert config.DEFAULT_MASTERY == 0.3, f"DEFAULT_MASTERY should be 0.3, got {config.DEFAULT_MASTERY}"
    assert config.FRESHNESS_RECENCY_DAYS == 3, f"FRESHNESS_RECENCY_DAYS should be 3"
    print(f"  LLM Provider: {config.LLM_PROVIDER}")
    print(f"  Google API Key: {config.GOOGLE_API_KEY[:8]}...")
    print(f"  Default Mastery: {config.DEFAULT_MASTERY}")
    print(f"  Freshness Recency: {config.FRESHNESS_RECENCY_DAYS} days")
    print("  PASSED\n")


def test_data_loading():
    print("=" * 60)
    print("TEST 2: Data Loading (videos + vector DB)")
    print("=" * 60)
    from data_ingestion import load_videos, get_collection

    videos = load_videos()
    assert len(videos) == 15, f"Expected 15 videos, got {len(videos)}"
    print(f"  Loaded {len(videos)} videos")

    # Check each video has required fields
    required_fields = ["video_id", "title", "topic", "transcript", "chunks", "duration"]
    for v in videos:
        for field in required_fields:
            assert field in v, f"Video {v.get('video_id', '?')} missing field: {field}"
        assert len(v["transcript"]) > 100, f"Video {v['video_id']} transcript too short: {len(v['transcript'])}"
        assert len(v["chunks"]) > 0, f"Video {v['video_id']} has no chunks"

    print(f"  All videos have required fields")

    # Test ChromaDB collection
    collection = get_collection()
    count = collection.count()
    assert count > 0, f"ChromaDB collection empty, expected > 0 chunks"
    print(f"  ChromaDB collection: {count} chunks indexed")

    # Test a query
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode("stock market investing shares").tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    assert results["documents"][0], "ChromaDB query returned no results"
    print(f"  Sample query returned {len(results['documents'][0])} results")
    print(f"    Top result topic: {results['metadatas'][0][0]['topic']}")
    print("  PASSED\n")

    return videos


# ══════════════════════════════════════════════════════════════
# TEST 3: Learner Profile (adaptive alpha, default mastery)
# ══════════════════════════════════════════════════════════════
def test_learner_profile():
    print("=" * 60)
    print("TEST 3: Learner Profile Management")
    print("=" * 60)
    import config
    from learner_profile import (
        load_profile, save_profile, record_watch, get_watched_video_ids,
        calculate_difficulty, calculate_quiz_length, get_weak_concepts,
        get_weak_topics, get_mastery_summary, get_due_recalls,
        create_default_profile, _adaptive_alpha,
    )

    profile = load_profile()
    assert profile["user_id"] == "user_001", f"Wrong user_id: {profile['user_id']}"
    assert len(profile["watch_history"]) > 0, "No watch history in synthetic profile"
    assert len(profile["quiz_history"]) > 0, "No quiz history in synthetic profile"
    print(f"  Profile loaded: {len(profile['watch_history'])} watches, {len(profile['quiz_history'])} quizzes")

    # Test adaptive alpha
    alpha_new = _adaptive_alpha({"quiz_history": []}, "new_topic")
    assert alpha_new == 0.6, f"Alpha for new topic should be 0.6, got {alpha_new}"
    alpha_6 = _adaptive_alpha({"quiz_history": [{"topic": "x"}] * 6}, "x")
    assert alpha_6 == 0.3, f"Alpha after 6 quizzes should be 0.3, got {alpha_6}"
    print(f"  Adaptive alpha: new topic={alpha_new}, after 6 quizzes={alpha_6}")

    # Test default mastery = 0.3 in difficulty calc
    fresh_profile = create_default_profile()
    diff, score = calculate_difficulty(fresh_profile, "unknown_topic")
    # With DEFAULT_MASTERY=0.3: topic_accuracy=0.3, related=0.3, attempt=0.7
    # score = 0.4*(1-0.3) + 0.3*(1-0.3) + 0.3*0.7 = 0.28+0.21+0.21 = 0.7 -> hard
    assert diff in ("easy", "medium", "hard"), f"Invalid difficulty: {diff}"
    print(f"  Fresh profile difficulty for unknown topic: {diff} (score={score:.3f})")

    # Test watched video IDs
    watched = get_watched_video_ids(profile)
    assert len(watched) > 0, "No watched video IDs"
    print(f"  Watched video IDs: {len(watched)}")

    # Test difficulty calculation
    difficulty, score = calculate_difficulty(profile, "stock market")
    assert difficulty in ("easy", "medium", "hard"), f"Invalid difficulty: {difficulty}"
    assert 0 <= score <= 1, f"Invalid difficulty score: {score}"
    print(f"  Difficulty for 'stock market': {difficulty} (score={score:.3f})")

    # Test quiz length
    length = calculate_quiz_length(profile)
    assert 3 <= length <= 8, f"Invalid quiz length: {length}"
    print(f"  Quiz length: {length}")

    # Test weak concepts/topics
    weak_concepts = get_weak_concepts(profile)
    weak_topics = get_weak_topics(profile)
    print(f"  Weak concepts: {len(weak_concepts)}")
    print(f"  Weak topics: {len(weak_topics)}")

    # Test mastery summary
    summary = get_mastery_summary(profile)
    assert "total_videos_watched" in summary
    assert "average_quiz_score" in summary
    print(f"  Mastery summary: avg score={summary['average_quiz_score']}, "
          f"recall accuracy={summary['recall_accuracy']}")

    print("  PASSED\n")
    return profile


# ══════════════════════════════════════════════════════════════
# TEST 4: LLM Interface
# ══════════════════════════════════════════════════════════════
def test_llm():
    print("=" * 60)
    print("TEST 4: LLM Interface (Gemini)")
    print("=" * 60)
    from llm import generate, generate_json

    # Test basic generation
    text = generate("What is 2+2? Reply with just the number.", max_tokens=50)
    assert text and len(text) > 0, "LLM returned empty response"
    print(f"  Basic generation: '{text.strip()[:50]}'")

    # Test JSON generation
    result = generate_json(
        'Return a JSON object with key "answer" and value 42. Only JSON, nothing else.',
        max_tokens=50
    )
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "answer" in result, f"Missing 'answer' key in: {result}"
    print(f"  JSON generation: {result}")

    # Test JSON array generation
    result = generate_json(
        'Return a JSON array with 3 strings: ["hello", "world", "test"]. Only JSON.',
        max_tokens=50
    )
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    print(f"  JSON array generation: {result}")

    print("  PASSED\n")


# ══════════════════════════════════════════════════════════════
# TEST 5: Recap Generator (dual retrieval)
# ══════════════════════════════════════════════════════════════
def test_recap(videos, profile):
    print("=" * 60)
    print("TEST 5: Personalized Recap Generator (dual retrieval)")
    print("=" * 60)
    from recap_generator import retrieve_relevant_past_context, generate_recap

    # Pick a video the user has watched
    watched_ids = set(w["video_id"] for w in profile["watch_history"])
    current_video = next(v for v in videos if v["video_id"] in watched_ids)
    print(f"  Testing with: {current_video['title'][:50]}")

    # Test dual retrieval
    contexts = retrieve_relevant_past_context(current_video["transcript"], profile, top_k=5)
    print(f"  Retrieved {len(contexts)} past contexts (dual query)")
    sources = set()
    for ctx in contexts:
        sources.add(ctx.get("retrieval_source", "unknown"))
        print(f"    - {ctx['title'][:40]} (sim={ctx['similarity']:.3f}, source={ctx.get('retrieval_source', '?')})")
    print(f"  Retrieval sources: {sources}")

    # Test recap generation (uses full transcript, no trimming)
    recap = generate_recap(current_video, profile)
    assert "bullets" in recap, f"Missing 'bullets' in recap: {recap.keys()}"
    assert len(recap["bullets"]) == 3, f"Expected 3 bullets, got {len(recap['bullets'])}"
    print(f"  Recap generated:")
    for i, b in enumerate(recap["bullets"], 1):
        print(f"    {i}. {b}")
    print(f"  Reasoning: {recap['reasoning'][:100]}")
    print(f"  Past contexts used: {len(recap['past_contexts_used'])}")

    print("  PASSED\n")


# ══════════════════════════════════════════════════════════════
# TEST 6: Quiz Engine (full transcript)
# ══════════════════════════════════════════════════════════════
def test_quiz(videos, profile):
    print("=" * 60)
    print("TEST 6: Adaptive Quiz Engine (full transcript)")
    print("=" * 60)
    from quiz_engine import (
        extract_concepts, generate_adaptive_quiz, process_answer,
        get_quiz_score, adapt_difficulty
    )

    # Pick a watched video
    watched_ids = set(w["video_id"] for w in profile["watch_history"])
    video = next(v for v in videos if v["video_id"] in watched_ids)
    print(f"  Testing with: {video['title'][:50]}")
    print(f"  Transcript length: {len(video['transcript'])} chars (full, no trimming)")

    # Test concept extraction
    concepts = extract_concepts(video)
    assert isinstance(concepts, list), f"Expected list of concepts, got {type(concepts)}"
    assert len(concepts) > 0, "No concepts extracted"
    print(f"  Extracted {len(concepts)} concepts:")
    for c in concepts[:3]:
        print(f"    - {c.get('concept', '?')}: {c.get('description', '?')[:50]}")

    # Test full quiz generation
    quiz = generate_adaptive_quiz(video, profile)
    assert "questions" in quiz, f"Missing 'questions' in quiz: {quiz.keys()}"
    assert len(quiz["questions"]) > 0, "No questions generated"
    assert "initial_difficulty" in quiz
    assert "quiz_length" in quiz
    print(f"  Quiz generated: {len(quiz['questions'])} questions, "
          f"difficulty={quiz['initial_difficulty']}, length={quiz['quiz_length']}")

    # Validate each question structure
    for i, q in enumerate(quiz["questions"]):
        assert "question" in q, f"Q{i} missing 'question'"
        assert "options" in q, f"Q{i} missing 'options'"
        assert "correct_answer" in q, f"Q{i} missing 'correct_answer'"
        assert q["correct_answer"] in ("A", "B", "C", "D"), f"Q{i} invalid answer: {q['correct_answer']}"
    print(f"  All questions have valid structure")

    # Simulate answering
    for i, q in enumerate(quiz["questions"]):
        user_ans = q["correct_answer"] if i < 2 else "A" if q["correct_answer"] != "A" else "B"
        result = process_answer(quiz, i, user_ans)
        if result["adaptation"]["adapted"]:
            print(f"  Adaptation at Q{i+1}: {result['adaptation']['old_difficulty']} -> "
                  f"{result['adaptation']['new_difficulty']}")

    # Test scoring
    score = get_quiz_score(quiz)
    assert "score" in score
    assert "accuracy" in score
    assert "concept_scores" in score
    print(f"  Score: {score['score']}/{score['total']} ({score['accuracy']:.0%})")
    print(f"  Concept breakdown: {list(score['concept_scores'].keys())}")

    # Test adapt_difficulty standalone
    assert adapt_difficulty([], "medium") == "medium"
    print(f"  adapt_difficulty edge case (empty): PASSED")

    print("  PASSED\n")
    return quiz, score


# ══════════════════════════════════════════════════════════════
# TEST 7: Recommender (per-concept queries, embedding overlap, freshness)
# ══════════════════════════════════════════════════════════════
def test_recommender(profile, quiz_score):
    print("=" * 60)
    print("TEST 7: Video Recommender (per-concept, embedding overlap, freshness)")
    print("=" * 60)
    from recommender import recommend_next_video, _compute_freshness, _embedding_topic_overlap, get_embed_model

    # Test freshness computation
    fresh_profile_no_quiz = {"quiz_history": []}
    assert _compute_freshness(fresh_profile_no_quiz, "anything") == 1.0, "No quiz = freshness 1.0"

    from datetime import datetime
    recent_profile = {"quiz_history": [{"topic": "test", "timestamp": datetime.now().isoformat()}]}
    assert _compute_freshness(recent_profile, "test") == 0.5, "Just quizzed = freshness 0.5"
    assert _compute_freshness(recent_profile, "other") == 1.0, "Different topic = freshness 1.0"
    print(f"  Freshness: no quiz=1.0, just quizzed=0.5, other topic=1.0")

    # Test embedding topic overlap
    model = get_embed_model()
    weak = [{"topic": "stock market"}, {"topic": "inflation"}]
    overlap_related = _embedding_topic_overlap(model, "investing in stocks", weak)
    overlap_unrelated = _embedding_topic_overlap(model, "photosynthesis", weak)
    assert overlap_related > overlap_unrelated, (
        f"Related topic should have higher overlap: {overlap_related:.3f} vs {overlap_unrelated:.3f}")
    print(f"  Embedding topic overlap: related={overlap_related:.3f}, unrelated={overlap_unrelated:.3f}")

    # Test recommendations
    recs = recommend_next_video(profile, quiz_score, top_n=3)
    assert isinstance(recs, list), f"Expected list, got {type(recs)}"
    print(f"  Got {len(recs)} recommendations:")

    watched_ids = set(w["video_id"] for w in profile["watch_history"])
    for r in recs:
        assert "video_id" in r
        assert "title" in r
        assert "score" in r
        assert "reason" in r
        assert "topic_overlap" in r
        assert "freshness" in r
        assert r["video_id"] not in watched_ids, f"Recommended already-watched video: {r['video_id']}"
        print(f"    - {r['title'][:40]}")
        print(f"      score={r['score']}, semantic={r['semantic_similarity']}, "
              f"overlap={r['topic_overlap']}, fresh={r['freshness']}")

    # Test without quiz result
    recs2 = recommend_next_video(profile, None, top_n=2)
    print(f"  Without quiz result: {len(recs2)} recommendations")

    print("  PASSED\n")


# ══════════════════════════════════════════════════════════════
# TEST 8: Recall Scheduler (structured rubric)
# ══════════════════════════════════════════════════════════════
def test_recall(videos, profile):
    print("=" * 60)
    print("TEST 8: Recall Scheduler (structured rubric)")
    print("=" * 60)
    from recall_scheduler import (
        find_weakest_concept_for_video, generate_recall_question,
        evaluate_recall_answer, schedule_recalls_for_video,
    )
    from learner_profile import get_due_recalls

    # Find weakest concept
    watched_id = profile["watch_history"][0]["video_id"]
    concept = find_weakest_concept_for_video(profile, watched_id)
    print(f"  Weakest concept for {watched_id}: {concept}")

    # Get the video
    video = next(v for v in videos if v["video_id"] == watched_id)

    # Generate recall question (uses full transcript now)
    question = generate_recall_question(video, concept or video["topic"], profile)
    assert "question" in question, f"Missing 'question': {question.keys()}"
    assert "expected_answer" in question, f"Missing 'expected_answer'"
    print(f"  Recall question: {question['question'][:80]}")
    print(f"  Expected answer: {question['expected_answer'][:80]}")

    # Evaluate with structured rubric
    eval_result = evaluate_recall_answer(question, question["expected_answer"][:50])
    assert "score" in eval_result, f"Missing 'score' in evaluation: {eval_result}"
    assert "feedback" in eval_result
    assert "factual_accuracy" in eval_result, "Missing rubric dimension: factual_accuracy"
    assert "completeness" in eval_result, "Missing rubric dimension: completeness"
    assert "key_term_usage" in eval_result, "Missing rubric dimension: key_term_usage"
    assert 0 <= eval_result["score"] <= 1, f"Invalid score: {eval_result['score']}"
    print(f"  Rubric scores: accuracy={eval_result['factual_accuracy']}, "
          f"completeness={eval_result['completeness']}, terms={eval_result['key_term_usage']}")
    print(f"  Overall: {eval_result['score']:.2f}, feedback={eval_result['feedback'][:60]}")

    # Evaluate a wrong answer
    eval_wrong = evaluate_recall_answer(question, "I have no idea")
    print(f"  Wrong answer eval: score={eval_wrong['score']:.2f}")

    # Schedule recall
    schedule_result = schedule_recalls_for_video(profile, watched_id)
    assert schedule_result["scheduled"], "Recall not scheduled"
    print(f"  Scheduled: {schedule_result['message']}")

    print("  PASSED\n")


# ══════════════════════════════════════════════════════════════
# TEST 9: Metrics
# ══════════════════════════════════════════════════════════════
def test_metrics(profile):
    print("=" * 60)
    print("TEST 9: Learning Metrics")
    print("=" * 60)
    from metrics import compute_metrics

    metrics = compute_metrics(profile)
    assert "summary" in metrics
    assert "quiz_scores" in metrics
    assert "difficulty_progression" in metrics
    assert "mastery_current" in metrics

    s = metrics["summary"]
    print(f"  Total videos watched: {s['total_videos_watched']}")
    print(f"  Total quizzes: {s['total_quizzes']}")
    print(f"  Overall avg score: {s['overall_avg_score']:.0%}")
    print(f"  Score improvement: {s['score_improvement']:+.0%}")
    print(f"  Recall accuracy: {s['recall_accuracy']:.0%}")
    print(f"  Topics mastered: {s['topics_mastered']}")
    print(f"  Topics weak: {s['topics_weak']}")

    print("  PASSED\n")


# ══════════════════════════════════════════════════════════════
# TEST 10: Full Pipeline (end-to-end)
# ══════════════════════════════════════════════════════════════
def test_full_pipeline(videos, profile):
    print("=" * 60)
    print("TEST 10: Full End-to-End Pipeline")
    print("=" * 60)
    from learner_profile import (
        record_watch, record_quiz_result, save_profile, get_mastery_summary
    )
    from recap_generator import generate_recap
    from quiz_engine import generate_adaptive_quiz, process_answer, get_quiz_score
    from recommender import recommend_next_video
    from recall_scheduler import schedule_recalls_for_video
    from metrics import compute_metrics

    # Pick an unwatched video
    watched_ids = set(w["video_id"] for w in profile["watch_history"])
    new_video = next(v for v in videos if v["video_id"] not in watched_ids)
    print(f"  New video: {new_video['title'][:50]}")

    # Step 1: Watch (with realistic completion rate)
    print("\n  STEP 1: Record watch (completion=85%)")
    profile = record_watch(profile, new_video["video_id"], completion_rate=0.85)
    print(f"    Watched. Total watches: {len(profile['watch_history'])}")

    # Step 2: Generate recap
    print("\n  STEP 2: Generate personalized recap")
    recap = generate_recap(new_video, profile)
    for i, b in enumerate(recap["bullets"], 1):
        print(f"    {i}. {b}")

    # Step 3: Take quiz
    print("\n  STEP 3: Generate and take adaptive quiz")
    quiz = generate_adaptive_quiz(new_video, profile)
    assert len(quiz["questions"]) > 0, "Quiz generation returned 0 questions (LLM may have failed)"
    print(f"    Generated {len(quiz['questions'])} questions at {quiz['initial_difficulty']} difficulty")

    # Simulate answering (mix of correct/wrong)
    import random
    for i, q in enumerate(quiz["questions"]):
        if random.random() < 0.6:
            user_ans = q["correct_answer"]
        else:
            wrong = [k for k in ("A", "B", "C", "D") if k != q["correct_answer"]]
            user_ans = random.choice(wrong)
        result = process_answer(quiz, i, user_ans)
        status = "correct" if result["correct"] else "wrong"
        adapt = f" [ADAPTED: {result['adaptation']['old_difficulty']}->{result['adaptation']['new_difficulty']}]" if result["adaptation"]["adapted"] else ""
        print(f"    Q{i+1}: {status}{adapt}")

    score = get_quiz_score(quiz)
    print(f"    Final score: {score['score']}/{score['total']} ({score['accuracy']:.0%})")

    # Record quiz result
    profile = record_quiz_result(
        profile, new_video["video_id"], new_video["topic"],
        quiz["questions"], score["accuracy"],
        quiz["current_difficulty"], score["completion_rate"],
    )

    # Step 4: Get recommendations
    print("\n  STEP 4: Get next video recommendations")
    recs = recommend_next_video(profile, score, top_n=3)
    for r in recs:
        print(f"    - {r['title'][:40]} (score={r['score']}, overlap={r['topic_overlap']:.2f}, fresh={r.get('freshness', 1.0)})")

    # Step 5: Schedule recall
    print("\n  STEP 5: Schedule recall check")
    recall_info = schedule_recalls_for_video(profile, new_video["video_id"], score)
    print(f"    {recall_info['message']}")

    # Step 6: Compute metrics
    print("\n  STEP 6: Compute learning metrics")
    metrics = compute_metrics(profile)
    s = metrics["summary"]
    print(f"    Avg score: {s['overall_avg_score']:.0%}")
    print(f"    Improvement: {s['score_improvement']:+.0%}")
    print(f"    Topics mastered: {s['topics_mastered']}, weak: {s['topics_weak']}")

    print("\n  FULL PIPELINE PASSED\n")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = {}
    tests = [
        ("config", test_config, []),
        ("data_loading", test_data_loading, []),
        ("learner_profile", test_learner_profile, []),
        ("llm", test_llm, []),
    ]

    # Run foundational tests first
    videos = None
    profile = None

    for name, func, args in tests:
        try:
            result = func(*args)
            results[name] = "PASSED"
            if name == "data_loading":
                videos = result
            elif name == "learner_profile":
                profile = result
        except Exception as e:
            results[name] = f"FAILED: {e}"
            print(f"  FAILED: {e}")
            traceback.print_exc()
            print()

    # Run tests that depend on videos/profile
    if videos and profile:
        dependent_tests = [
            ("recap", test_recap, [videos, profile]),
            ("quiz", test_quiz, [videos, profile]),
        ]

        quiz_score = None
        for name, func, args in dependent_tests:
            try:
                result = func(*args)
                results[name] = "PASSED"
                if name == "quiz":
                    _, quiz_score = result
            except Exception as e:
                results[name] = f"FAILED: {e}"
                print(f"  FAILED: {e}")
                traceback.print_exc()
                print()

        # Recommender needs quiz_score
        try:
            test_recommender(profile, quiz_score)
            results["recommender"] = "PASSED"
        except Exception as e:
            results["recommender"] = f"FAILED: {e}"
            print(f"  FAILED: {e}")
            traceback.print_exc()
            print()

        # Recall
        try:
            test_recall(videos, profile)
            results["recall"] = "PASSED"
        except Exception as e:
            results["recall"] = f"FAILED: {e}"
            print(f"  FAILED: {e}")
            traceback.print_exc()
            print()

        # Metrics
        try:
            test_metrics(profile)
            results["metrics"] = "PASSED"
        except Exception as e:
            results["metrics"] = f"FAILED: {e}"
            print(f"  FAILED: {e}")
            traceback.print_exc()
            print()

        # Full pipeline (only if all previous passed)
        all_passed = all(v == "PASSED" for v in results.values())
        if all_passed:
            try:
                test_full_pipeline(videos, profile)
                results["full_pipeline"] = "PASSED"
            except Exception as e:
                results["full_pipeline"] = f"FAILED: {e}"
                print(f"  FAILED: {e}")
                traceback.print_exc()
                print()
        else:
            results["full_pipeline"] = "SKIPPED (earlier failures)"

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        icon = "PASS" if status == "PASSED" else "FAIL" if "FAILED" in status else "SKIP"
        print(f"  [{icon}] {name}: {status}")

    failed = sum(1 for v in results.values() if "FAILED" in v)
    print(f"\n  {len(results) - failed}/{len(results)} tests passed")
    sys.exit(1 if failed else 0)
