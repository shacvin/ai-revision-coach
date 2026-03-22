"""Streamlit UI for the AI Revision Coach."""

import streamlit as st
import json

from data_ingestion import load_videos, get_collection
from learner_profile import (
    load_profile, save_profile, create_default_profile,
    record_watch, record_quiz_result,
    get_mastery_summary, get_watched_video_ids,
    record_recall, get_due_recalls, get_weak_concepts, get_weak_topics,
)
from recap_generator import generate_recap
from quiz_engine import (
    generate_adaptive_quiz, process_answer, get_quiz_score,
    generate_additional_questions,
)
from recommender import recommend_next_video
from recall_scheduler import (
    generate_recall_question, evaluate_recall_answer,
    schedule_recalls_for_video, find_weakest_concept_for_video,
)
from metrics import compute_metrics
from tts import synthesize_recap

# ── Page config ──────────────────────────────────────────────────

st.set_page_config(page_title="AI Revision Coach", page_icon="🎓", layout="wide")

# ── Session state init ───────────────────────────────────────────

if "videos" not in st.session_state:
    try:
        st.session_state.videos = load_videos()
        st.session_state.collection = get_collection()
    except Exception as e:
        st.error(f"Failed to load data: {e}. Run `python build_dataset.py` first.")
        st.stop()

if "profile" not in st.session_state:
    st.session_state.profile = load_profile()

if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "quiz_idx" not in st.session_state:
    st.session_state.quiz_idx = 0
if "quiz_done" not in st.session_state:
    st.session_state.quiz_done = False
if "quiz_video" not in st.session_state:
    st.session_state.quiz_video = None
if "quiz_saved" not in st.session_state:
    st.session_state.quiz_saved = False
if "recap" not in st.session_state:
    st.session_state.recap = None
if "recap_video_id" not in st.session_state:
    st.session_state.recap_video_id = None


def reload_profile():
    st.session_state.profile = load_profile()


# ── Sidebar: profile management ─────────────────────────────────

with st.sidebar:
    st.title("🎓 AI Revision Coach")
    st.divider()

    profile = st.session_state.profile
    summary = get_mastery_summary(profile)

    st.metric("Videos Watched", summary["total_videos_watched"])
    st.metric("Quizzes Taken", summary["total_quizzes_taken"])
    if summary["total_quizzes_taken"]:
        st.metric("Avg Quiz Score", f"{summary['average_quiz_score']:.0%}")
    if summary["pending_recalls"]:
        st.warning(f"📋 {summary['pending_recalls']} recall(s) due!")

    st.divider()
    st.subheader("Profile Management")

    # Demo helper — make all pending recalls due today
    if profile.get("recall_queue"):
        pending = len(profile["recall_queue"])
        if st.button(f"⏰ Force {pending} Recall(s) Due Now", help="Demo: sets all recall dates to today"):
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            for r in st.session_state.profile["recall_queue"]:
                r["scheduled_date"] = today
            save_profile(st.session_state.profile)
            st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Fresh", help="Blank slate"):
            p = create_default_profile("user_001")
            save_profile(p)
            st.session_state.profile = p
            st.session_state.quiz = None
            st.rerun()
    with col2:
        if st.button("Quick", help="3 videos watched"):
            from synthetic_data import generate_synthetic_profile
            p = generate_synthetic_profile(st.session_state.videos, num_watched=3)
            from datetime import datetime
            from learner_profile import get_watched_video_ids as gwi
            watched = gwi(p)
            if watched and p.get("concept_mastery"):
                weakest = min(p["concept_mastery"].items(), key=lambda x: x[1])
                today = datetime.now().strftime("%Y-%m-%d")
                p["recall_queue"].append({
                    "video_id": watched[0], "concept": weakest[0],
                    "scheduled_date": today,
                })
                save_profile(p)
            st.session_state.profile = p
            st.session_state.quiz = None
            st.rerun()
    with col3:
        if st.button("Full", help="6 videos watched"):
            from synthetic_data import generate_synthetic_profile
            p = generate_synthetic_profile(st.session_state.videos, num_watched=6)
            st.session_state.profile = p
            st.session_state.quiz = None
            st.rerun()


# ── Main tabs ────────────────────────────────────────────────────

tab_watch, tab_quiz, tab_recs, tab_recall, tab_metrics = st.tabs([
    "📺 Watch & Recap", "📝 Adaptive Quiz", "🎯 Recommendations",
    "🧠 Recall Check", "📊 Learning Dashboard",
])

# ──────────────────────────────────────────────────────────────────
# TAB 1: WATCH VIDEO & RECAP
# ──────────────────────────────────────────────────────────────────

with tab_watch:
    st.header("Watch a Video & Get Personalized Recap")

    profile = st.session_state.profile
    videos = st.session_state.videos
    watched_ids = set(get_watched_video_ids(profile))

    video_options = {
        f"{'✅' if v['video_id'] in watched_ids else '⬜'} {v['title']} [{v['topic']}]": v
        for v in videos
    }

    selected_label = st.selectbox("Select a video", list(video_options.keys()))
    video = video_options[selected_label]

    # Clear cached recap when a different video is selected
    if st.session_state.recap_video_id != video["video_id"]:
        st.session_state.recap = None
        st.session_state.recap_video_id = video["video_id"]

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.caption(f"**Topic:** {video['topic']}  |  **Duration:** {video['duration'] // 60}m{video['duration'] % 60:02d}s")
        st.markdown(f"[▶ Watch on YouTube](https://www.youtube.com/watch?v={video['video_id']})")
    with col_b:
        completion = st.slider("How much did you watch?", 0, 100, 90, 5, format="%d%%")

    with st.expander("📜 Transcript preview"):
        st.text(video["transcript"][:800] + "...")

    if st.button("✅ Mark as Watched & Generate Recap", type="primary"):
        profile = record_watch(profile, video["video_id"], completion_rate=completion / 100)
        st.session_state.profile = profile

        with st.spinner("Generating personalized recap..."):
            recap = generate_recap(video, profile)

        with st.spinner("Generating audio recap..."):
            audio_bytes = synthesize_recap(recap["bullets"])
        recap["audio_bytes"] = audio_bytes

        st.session_state.recap = recap
        st.rerun()

    # Display recap from session state (persists across reruns)
    if st.session_state.recap:
        recap = st.session_state.recap
        st.success("✅ Video recorded and recap generated!")
        st.subheader("Your 60-Second Recap")
        for i, bullet in enumerate(recap["bullets"], 1):
            st.markdown(f"**{i}.** {bullet}")

        if recap.get("audio_bytes"):
            st.audio(recap["audio_bytes"], format="audio/wav")

        st.caption(f"*Personalization:* {recap['reasoning']}")
        if recap["past_contexts_used"]:
            refs = ", ".join(c["title"][:30] for c in recap["past_contexts_used"][:3])
            st.caption(f"*Referenced past videos:* {refs}")

# ──────────────────────────────────────────────────────────────────
# TAB 2: ADAPTIVE QUIZ
# ──────────────────────────────────────────────────────────────────

with tab_quiz:
    st.header("Adaptive Quiz")

    profile = st.session_state.profile
    watched_ids = set(get_watched_video_ids(profile))
    quiz_videos = [v for v in st.session_state.videos if v["video_id"] in watched_ids]

    if not quiz_videos:
        st.info("Watch a video first to take a quiz.")
    else:
        quiz_options = {f"{v['title']} [{v['topic']}]": v for v in quiz_videos}
        selected_quiz_label = st.selectbox("Select video for quiz", list(quiz_options.keys()), key="quiz_select")
        selected_quiz_video = quiz_options[selected_quiz_label]

        # Generate quiz button
        if st.button("🚀 Generate Quiz", type="primary"):
            st.session_state.quiz_video = selected_quiz_video
            st.session_state.quiz_done = False
            st.session_state.quiz_idx = 0
            st.session_state.quiz_saved = False
            with st.spinner("Generating adaptive quiz..."):
                quiz = generate_adaptive_quiz(selected_quiz_video, profile)
            st.session_state.quiz = quiz
            st.rerun()

        quiz = st.session_state.quiz

        if quiz and not st.session_state.quiz_done:
            st.markdown(f"**Difficulty:** {quiz['initial_difficulty'].upper()}  |  "
                        f"**Questions:** {len(quiz['questions'])}  |  "
                        f"**Score threshold:** {quiz['difficulty_score']}")

            idx = st.session_state.quiz_idx

            if idx < len(quiz["questions"]):
                q = quiz["questions"][idx]
                st.subheader(f"Question {idx + 1} / {len(quiz['questions'])}")
                st.caption(f"Difficulty: {q.get('difficulty', quiz['current_difficulty']).upper()}  |  Concept: {q.get('concept', '?')}")
                st.markdown(f"**{q['question']}**")

                options = q["options"]
                answer = st.radio("Your answer:", list(options.keys()),
                                  format_func=lambda k: f"{k}. {options[k]}",
                                  key=f"q_{idx}")

                if st.button("Submit Answer", key=f"submit_{idx}"):
                    result = process_answer(quiz, idx, answer)

                    if result["correct"]:
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Wrong. Correct: **{result['correct_answer']}**")
                        if result.get("explanation"):
                            st.info(f"💡 {result['explanation']}")

                    # Show adaptation
                    if result["adaptation"]["adapted"]:
                        old = result["adaptation"]["old_difficulty"]
                        new = result["adaptation"]["new_difficulty"]
                        st.warning(f"⚡ Difficulty adapted: {old.upper()} → {new.upper()}")
                        # Regenerate remaining questions at new difficulty
                        remaining = len(quiz["questions"]) - (idx + 1)
                        if remaining > 0:
                            missed = [q2["concept"] for q2 in quiz["questions"][:idx + 1]
                                      if not q2.get("user_correct")]
                            with st.spinner(f"Regenerating {remaining} question(s) at {new.upper()} difficulty..."):
                                new_qs = generate_additional_questions(
                                    st.session_state.quiz_video, quiz["concepts"],
                                    new, remaining, profile,
                                    missed_concepts=missed if missed else None,
                                )
                            if new_qs:
                                quiz["questions"] = quiz["questions"][:idx + 1] + new_qs

                    st.session_state.quiz_idx = idx + 1
                    if st.session_state.quiz_idx >= len(quiz["questions"]):
                        st.session_state.quiz_done = True
                    st.rerun()

        if quiz and st.session_state.quiz_done:
            score = get_quiz_score(quiz)
            st.subheader("Quiz Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("Score", f"{score['score']}/{score['total']}")
            col2.metric("Accuracy", f"{score['accuracy']:.0%}")
            col3.metric("Final Difficulty", quiz["current_difficulty"].upper())

            if score["concept_scores"]:
                st.markdown("**Per-concept breakdown:**")
                for concept, cs in score["concept_scores"].items():
                    acc = cs["accuracy"]
                    icon = "🟢" if acc >= 0.7 else "🟡" if acc >= 0.4 else "🔴"
                    st.markdown(f"{icon} **{concept}**: {cs['correct']}/{cs['total']}")

            # Record result exactly once
            if not st.session_state.quiz_saved:
                video = st.session_state.quiz_video
                profile = record_quiz_result(
                    profile, video["video_id"], video["topic"],
                    quiz["questions"], score["accuracy"],
                    quiz["current_difficulty"], score["completion_rate"],
                )
                recall_info = schedule_recalls_for_video(profile, video["video_id"], score)
                st.session_state.profile = profile
                st.session_state.quiz_saved = True
                st.success(f"Results saved. {recall_info['message']}")
            else:
                st.success("Results already saved.")

            # Full question review
            st.divider()
            st.subheader("Question Review")
            for i, q in enumerate(quiz["questions"]):
                answered = q.get("user_answer") is not None
                if not answered:
                    continue
                correct = q.get("user_correct")
                icon = "✅" if correct else "❌"
                concept_label = f"*[{q.get('concept', '?')}]*"
                with st.expander(f"{icon} Q{i+1}: {q['question'][:80]}... — {concept_label}"):
                    opts = q.get("options", {})
                    for key, val in opts.items():
                        if key == q["correct_answer"] and key == q.get("user_answer"):
                            st.markdown(f"**{key}. {val}** ✅ ← your answer (correct)")
                        elif key == q["correct_answer"]:
                            st.markdown(f"**{key}. {val}** ✅ ← correct answer")
                        elif key == q.get("user_answer"):
                            st.markdown(f"~~{key}. {val}~~ ❌ ← your answer")
                        else:
                            st.markdown(f"{key}. {val}")
                    if q.get("explanation"):
                        st.info(f"💡 {q['explanation']}")

            if st.button("Take Another Quiz"):
                st.session_state.quiz = None
                st.session_state.quiz_done = False
                st.session_state.quiz_idx = 0
                st.session_state.quiz_saved = False
                st.rerun()


# ──────────────────────────────────────────────────────────────────
# TAB 3: RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────

with tab_recs:
    st.header("Video Recommendations")

    profile = st.session_state.profile

    if not profile["quiz_history"]:
        st.info("Take at least one quiz to get personalized recommendations.")
    else:
        if st.button("🔍 Get Recommendations", type="primary"):
            with st.spinner("Analyzing your weak areas..."):
                recs = recommend_next_video(profile, top_n=3)

            if not recs:
                st.success("You've covered all available content!")
            else:
                for i, rec in enumerate(recs, 1):
                    with st.container():
                        st.subheader(f"#{i}  {rec['title']}")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Match", f"{rec['score']:.0%}")
                        col2.metric("Semantic", f"{rec['semantic_similarity']:.2f}")
                        col3.metric("Topic Overlap", f"{rec['topic_overlap']:.2f}")
                        col4.metric("Freshness", f"{rec.get('freshness', 1.0):.1f}")
                        st.caption(rec["reason"])
                        if rec.get("weak_concepts_addressed"):
                            st.markdown(f"*Addresses:* {', '.join(rec['weak_concepts_addressed'])}")
                        st.divider()


# ──────────────────────────────────────────────────────────────────
# TAB 4: RECALL CHECK
# ──────────────────────────────────────────────────────────────────

with tab_recall:
    st.header("Spaced Repetition Recall Check")

    profile = st.session_state.profile
    due = get_due_recalls(profile)

    if not due:
        st.info("No recall checks due today.")
        if profile.get("recall_queue"):
            st.markdown("**Upcoming recalls:**")
            for r in profile["recall_queue"][:5]:
                st.markdown(f"- **{r['concept']}** (scheduled: {r['scheduled_date']})")
    else:
        video_map = {v["video_id"]: v for v in st.session_state.videos}
        st.markdown(f"You have **{len(due)}** recall check(s) due.")

        # Process one recall at a time
        if "recall_current" not in st.session_state:
            st.session_state.recall_current = 0
        if "recall_question" not in st.session_state:
            st.session_state.recall_question = None

        current = st.session_state.recall_current

        if current < len(due):
            recall_item = due[current]
            video = video_map.get(recall_item["video_id"])

            if video:
                st.subheader(f"Recall #{current + 1}: {recall_item['concept']}")
                st.caption(f"From: {video['title']}")

                # Generate question if not yet generated
                if st.session_state.recall_question is None:
                    with st.spinner("Generating recall question..."):
                        q = generate_recall_question(video, recall_item["concept"], profile)
                    st.session_state.recall_question = q

                question = st.session_state.recall_question
                st.markdown(f"**{question['question']}**")
                st.caption(f"Type: {question.get('question_type', 'short_answer')}")

                user_answer = st.text_area("Your answer:", key=f"recall_{current}")

                if st.button("Submit Recall Answer", key=f"recall_submit_{current}"):
                    if not user_answer.strip():
                        st.warning("Please type an answer.")
                    else:
                        with st.spinner("Evaluating..."):
                            evaluation = evaluate_recall_answer(question, user_answer)

                        score = evaluation["score"]
                        if score >= 0.7:
                            st.success(f"Great recall! Score: {score:.0%}")
                        elif score >= 0.4:
                            st.warning(f"Partial recall. Score: {score:.0%}")
                        else:
                            st.error(f"Needs review. Score: {score:.0%}")

                        # Show rubric breakdown
                        cols = st.columns(3)
                        cols[0].metric("Factual Accuracy", f"{evaluation.get('factual_accuracy', 0):.0%}")
                        cols[1].metric("Completeness", f"{evaluation.get('completeness', 0):.0%}")
                        cols[2].metric("Key Terms", f"{evaluation.get('key_term_usage', 0):.0%}")

                        st.markdown(f"**Feedback:** {evaluation['feedback']}")
                        st.markdown(f"**Key point:** {evaluation['correct_answer_summary']}")

                        profile = record_recall(
                            profile, video["video_id"], recall_item["concept"],
                            question["question"], user_answer, score
                        )
                        st.session_state.profile = profile

                        st.session_state.recall_current = current + 1
                        st.session_state.recall_question = None
                        st.rerun()
        else:
            st.success("All recall checks completed!")
            st.session_state.recall_current = 0
            st.session_state.recall_question = None


# ──────────────────────────────────────────────────────────────────
# TAB 5: LEARNING DASHBOARD
# ──────────────────────────────────────────────────────────────────

with tab_metrics:
    st.header("Learning Dashboard")

    profile = st.session_state.profile

    if not profile["quiz_history"]:
        st.info("Complete some quizzes to see your learning metrics.")
    else:
        metrics = compute_metrics(profile)
        s = metrics["summary"]

        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Videos Watched", s["total_videos_watched"])
        c2.metric("Avg Score", f"{s['overall_avg_score']:.0%}")
        c3.metric("Improvement", f"{s['score_improvement']:+.0%}")
        c4.metric("Recall Accuracy", f"{s['recall_accuracy']:.0%}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Quizzes", s["total_quizzes"])
        c6.metric("Completion", f"{s['avg_completion_rate']:.0%}")
        c7.metric("Topics Mastered", s["topics_mastered"])
        c8.metric("Topics Weak", s["topics_weak"])

        st.divider()
        col_left, col_right = st.columns(2)

        # Topic Mastery — coarse level, one score per video topic (e.g. "stock market")
        with col_left:
            if metrics["mastery_current"]:
                st.subheader("Topic Mastery")
                st.caption("One score per topic, updated after each quiz on that topic.")
                for topic, score in sorted(metrics["mastery_current"].items(), key=lambda x: x[1], reverse=True):
                    icon = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
                    st.progress(min(score, 1.0), text=f"{icon} {topic}  {score:.0%}")

        # Weakest Concepts — fine-grained, one score per concept within a topic
        with col_right:
            if metrics["concept_mastery"]:
                st.subheader("Weakest Concepts")
                st.caption("Per-question concepts extracted from the video. Shows bottom 8.")
                sorted_concepts = sorted(metrics["concept_mastery"].items(), key=lambda x: x[1])
                for concept, score in sorted_concepts[:8]:
                    icon = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
                    st.progress(min(score, 1.0), text=f"{icon} {concept}  {score:.0%}")

        st.divider()

        # Quiz score progression
        if metrics["quiz_scores"]:
            st.subheader("Quiz Score Progression")
            import pandas as pd
            df = pd.DataFrame(metrics["quiz_scores"])
            st.line_chart(df.set_index("quiz_number")["score"])

        col3, col4 = st.columns(2)

        # Difficulty history
        with col3:
            if metrics["difficulty_progression"]:
                st.subheader("Difficulty per Quiz")
                for d in metrics["difficulty_progression"]:
                    diff = d["difficulty_label"]
                    icon = "🟢" if diff == "easy" else "🟡" if diff == "medium" else "🔴"
                    st.markdown(f"Quiz {d['quiz_number']} {icon} **{diff}** — {d['topic']}")

        # Recall log
        with col4:
            rm = metrics["recall_metrics"]
            if rm["total_attempts"] > 0:
                st.subheader(f"Recall Log ({rm['correct']}/{rm['total_attempts']} correct)")
                for r in rm["by_attempt"]:
                    icon = "✅" if r["correct"] else "❌"
                    st.markdown(f"{icon} **{r['concept']}** ({r['score']:.0%})")

        # Score improvement over time (only if ≥2 quizzes on same topic)
        if metrics["recommendation_effectiveness"]:
            st.subheader("Score Improvement by Topic")
            for topic, eff in metrics["recommendation_effectiveness"].items():
                icon = "📈" if eff["improved"] else "📉"
                st.markdown(f"{icon} **{topic}**: {eff['first_score']:.0%} → {eff['last_score']:.0%} ({eff['delta']:+.0%})")
