"""Parse pasted YouTube transcripts and build the video dataset + vector index."""

import json
import os
import re

import config

# Video metadata (verified: all < 8 min, all educational)
VIDEOS = [
    {"video_id": "p7HKvqRI_Bo", "title": "How does the stock market work? - Oliver Elfenbaum",
     "topic": "stock market", "tags": ["stock market", "investing", "TED-Ed", "finance", "shares"],
     "duration": 270, "channel": "TED-Ed"},
    {"video_id": "F3QpgXBtDeo", "title": "How The Stock Exchange Works (For Dummies)",
     "topic": "stock exchange", "tags": ["stock exchange", "trading", "finance", "Kurzgesagt"],
     "duration": 214, "channel": "Kurzgesagt"},
    {"video_id": "LwLh6ax0zTE", "title": "Demand and Supply Explained - Macro Topic 1.4",
     "topic": "supply and demand", "tags": ["supply", "demand", "economics", "macro", "equilibrium"],
     "duration": 403, "channel": "Jacob Clifford"},
    {"video_id": "wf91rEGw88Q", "title": "What Is Compound Interest?",
     "topic": "compound interest", "tags": ["compound interest", "finance", "investing", "Investopedia"],
     "duration": 120, "channel": "Investopedia"},
    {"video_id": "SwaCg7Gwtzw", "title": "What causes an economic recession? - Richard Coffin",
     "topic": "recession", "tags": ["recession", "economics", "TED-Ed", "business cycle"],
     "duration": 304, "channel": "TED-Ed"},
    {"video_id": "Hw9_DC-5OUM", "title": "INFLATION, Explained in 5 Minutes",
     "topic": "inflation", "tags": ["inflation", "economics", "prices", "monetary policy"],
     "duration": 298, "channel": "Casual Economics"},
    {"video_id": "Saqn77p63cE", "title": "What are government bonds?",
     "topic": "bonds", "tags": ["bonds", "government", "finance", "investing", "debt"],
     "duration": 157, "channel": "IG"},
    {"video_id": "URUJD5NEXC8", "title": "Biology: Cell Structure I Nucleus Medical Media",
     "topic": "cell structure", "tags": ["cell", "biology", "organelles", "nucleus", "membrane"],
     "duration": 441, "channel": "Nucleus Medical Media"},
    {"video_id": "TNKWgcFPHqw", "title": "DNA replication - 3D",
     "topic": "dna replication", "tags": ["DNA", "replication", "biology", "genetics", "helicase"],
     "duration": 208, "channel": "yourgenome"},
    {"video_id": "_lNF3_30lUE", "title": "How Small Is An Atom? Spoiler: Very Small.",
     "topic": "atomic structure", "tags": ["atom", "physics", "Kurzgesagt", "quantum", "size"],
     "duration": 297, "channel": "Kurzgesagt"},
    {"video_id": "mZt1Gn0R22Q", "title": "Myths and misconceptions about evolution - Alex Gendler",
     "topic": "natural selection", "tags": ["evolution", "natural selection", "TED-Ed", "biology"],
     "duration": 262, "channel": "TED-Ed"},
    {"video_id": "6hfOvs8pY1k", "title": "What's an algorithm? - David J. Malan",
     "topic": "algorithms", "tags": ["algorithm", "computer science", "TED-Ed", "programming"],
     "duration": 297, "channel": "TED-Ed"},
    {"video_id": "JUtes-k-VX4", "title": "Investing Basics: Mutual Funds",
     "topic": "mutual funds", "tags": ["mutual funds", "investing", "finance", "diversification"],
     "duration": 305, "channel": "TD Ameritrade"},
    {"video_id": "7_LPdttKXPc", "title": "How the Internet Works in 5 Minutes",
     "topic": "internet", "tags": ["internet", "networking", "technology", "TCP/IP"],
     "duration": 288, "channel": "Aaron"},
    {"video_id": "dQCsA2cCdvA", "title": "Biology Overview",
     "topic": "biology overview", "tags": ["biology", "life", "cells", "evolution", "ecology"],
     "duration": 273, "channel": "Bozeman Science"},
]

TRANSCRIPT_DIR = "./transcripts"

# Timestamp patterns
# Format 1 (normal): "0:06" on its own line
TS_LINE_PATTERN = re.compile(r'^\d{1,2}:\d{2}$')
# Format 2 (Hw9_DC-5OUM): "0:00have you noticed..." or "1:031 minute, 3 seconds..."
TS_INLINE_PATTERN = re.compile(r'^\d{1,2}:\d{2}(?:\d+\s+(?:seconds?|minutes?)(?:,\s*\d+\s*(?:seconds?|minutes?))?)?')


def infer_topic(transcript: str, existing_topics: list[str]) -> str:
    """Use LLM to assign a normalized topic label from transcript text.

    Re-uses an existing topic string if the content matches, so mastery
    scores stay consistent across videos on the same subject.
    """
    prompt = f"""You are categorizing an educational video by topic.

TRANSCRIPT:
{transcript}

EXISTING TOPICS ALREADY IN THE SYSTEM:
{json.dumps(existing_topics)}

Rules:
- If this video clearly covers one of the existing topics above, return that exact string.
- Otherwise invent a short new topic label: 2-5 words, all lowercase, no punctuation.

Return ONLY the topic string. No explanation. Examples: "stock market", "machine learning", "photosynthesis"
"""
    try:
        from llm import generate
        topic = generate(prompt, max_tokens=20).strip().lower().strip('"\'').strip()
        # Collapse any accidental newlines or extra spaces
        topic = " ".join(topic.split())
        return topic or "general"
    except Exception:
        return "general"


def parse_transcript(filepath: str) -> str:
    """Parse a pasted YouTube transcript into clean text, handling both formats."""
    with open(filepath) as f:
        raw = f.read()

    lines = raw.strip().split('\n')
    if not lines:
        return ""

    # Detect format: check if first line has timestamp glued to text (no newline separation)
    # Format 1 (normal): line is ONLY a timestamp like "0:06", next line is text
    # Format 2 (inline): line starts with timestamp then text immediately, e.g. "0:00have you..."
    sample = lines[0].strip()
    is_inline = bool(re.match(r'^\d{1,2}:\d{2}', sample)) and not TS_LINE_PATTERN.match(sample)

    text_parts = []

    if is_inline:
        # Format 2: "0:00have you noticed..." or "0:088 secondsinflation..."
        # Pattern: timestamp like "0:00" or "1:03" followed optionally by
        # duration like "8 seconds" or "1 minute, 3 seconds" then the actual text
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove leading "D:DD" timestamp
            cleaned = re.sub(r'^\d{1,2}:\d{2}', '', line).strip()
            # Remove leading duration like "8 seconds" or "1 minute, 37 seconds"
            cleaned = re.sub(
                r'^\d+\s+(?:seconds?|minutes?)(?:,\s*\d+\s*(?:seconds?|minutes?))?',
                '', cleaned
            ).strip()
            if cleaned:
                text_parts.append(cleaned)
    else:
        # Format 1: alternating timestamp lines and text lines
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip pure timestamp lines
            if TS_LINE_PATTERN.match(line):
                continue
            text_parts.append(line)

    return ' '.join(text_parts)


def chunk_text(text: str, target_size: int = 1200, overlap_sentences: int = 2) -> list[str]:
    """Recursive sentence-boundary chunking with overlap.

    Splits on sentence endings, groups into chunks of ~target_size chars,
    and overlaps by carrying the last N sentences into the next chunk.
    Falls back to character-based chunking with overlap if the text lacks
    sentence-ending punctuation (common in YouTube auto-transcripts).
    """
    # Split into sentences on '. ', '! ', '? ' boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Fallback: if sentence splitting yields too few splits for the text length,
    # the transcript likely lacks punctuation. Use character-based chunking.
    if len(sentences) <= 2 and len(text) > target_size:
        overlap = 200
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + target_size, len(text))
            chunk = text[start:end]
            # Try to break at a natural boundary (comma, space near end)
            if end < len(text):
                for sep in ['. ', ', ', ' ']:
                    last = chunk.rfind(sep)
                    if last > target_size // 2:
                        end = start + last + len(sep)
                        chunk = text[start:end]
                        break
            chunks.append(chunk.strip())
            # If we reached the end, stop. Otherwise advance with overlap.
            if end >= len(text):
                break
            start = end - overlap
        return [c for c in chunks if len(c) > 50]

    if not sentences:
        return [text] if len(text) > 50 else []

    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sent_len = len(sentence)
        if current_size + sent_len > target_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences else []
            current_size = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_size += sent_len

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return [c for c in chunks if len(c) > 50]


def build_dataset():
    """Build the full video dataset from pasted transcripts."""
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Collect topics that are already explicitly set, for LLM context
    existing_topics = [v["topic"] for v in VIDEOS if v.get("topic")]

    videos = []
    for video in VIDEOS:
        filepath = os.path.join(TRANSCRIPT_DIR, f"{video['video_id']}.txt")
        if not os.path.exists(filepath):
            print(f"MISSING: {filepath}")
            continue

        transcript = parse_transcript(filepath)
        if not transcript:
            print(f"EMPTY: {video['video_id']} - {video['title']}")
            continue

        # Auto-infer topic from transcript if not hardcoded
        if not video.get("topic"):
            print(f"  Inferring topic for {video['video_id']}...")
            video["topic"] = infer_topic(transcript, existing_topics)
            existing_topics.append(video["topic"])
            print(f"  → topic: \"{video['topic']}\"")

        video["url"] = f"https://www.youtube.com/watch?v={video['video_id']}"
        video["transcript"] = transcript
        video["chunks"] = chunk_text(transcript)
        videos.append(video)
        print(f"[{len(videos):2d}] {video['video_id']} | {len(transcript):5d} chars | "
              f"{len(video['chunks']):2d} chunks | {video['title'][:50]}")

    with open(config.VIDEOS_FILE, "w") as f:
        json.dump(videos, f, indent=2)
    print(f"\nSaved {len(videos)} videos to {config.VIDEOS_FILE}")
    return videos


if __name__ == "__main__":
    videos = build_dataset()
    if videos:
        from data_ingestion import build_vector_index
        build_vector_index(videos)
        print("\nDone! Dataset ready.")

        # Also generate synthetic learner profile
        from synthetic_data import generate_synthetic_profile
        profile = generate_synthetic_profile(videos, num_watched=6)
        print(f"\nGenerated learner profile: {len(profile['watch_history'])} videos watched")
        print(f"Mastery: {json.dumps(profile['mastery_scores'], indent=2)}")
