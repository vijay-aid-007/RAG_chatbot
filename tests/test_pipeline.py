"""
tests/test_pipeline.py
======================
Unit tests for all pipeline components.
Run with:  pytest tests/ -v

Tests are designed to run WITHOUT requiring the full dataset or API keys.
They use small synthetic fixtures.
"""

import json
import sys
import numpy as np
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

FAKE_MESSAGES = [
    {"msg_id": i, "conv_id": 1, "turn_in_conv": i, "speaker": "User 1" if i % 2 else "User 2", "text": text}
    for i, text in enumerate([
        "I love hiking in the mountains every weekend.",
        "That sounds amazing! I prefer rock climbing myself.",
        "Do you cook? I make pasta every Sunday.",
        "Yes! I cook Italian food for my family.",
        "I have two cats named Luna and Mochi.",
        "Pets are great companions. I have a golden retriever.",
        "I work as a software engineer at a startup.",
        "Interesting! I'm a nurse. Long shifts but rewarding.",
        "What music do you like? I listen to jazz.",
        "I'm into indie rock. Been to three concerts this year.",
    ], start=1)
]


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParser:
    def test_parse_conversation_basic(self):
        from core.parser import _parse_conversation
        raw = "User 1: Hello there\nUser 2: Hi! How are you?\nUser 1: Great thanks"
        msgs = _parse_conversation(raw, conv_id=1)
        assert len(msgs) == 3
        assert msgs[0]["speaker"] == "User 1"
        assert msgs[0]["text"] == "Hello there"
        assert msgs[2]["turn_in_conv"] == 3

    def test_parse_conversation_multiline(self):
        from core.parser import _parse_conversation
        raw = "User 1: Hello,\nhow are you?\nUser 2: Fine thanks!"
        msgs = _parse_conversation(raw, conv_id=1)
        # "how are you?" should be appended to User 1's message
        assert len(msgs) == 2
        assert "how are you?" in msgs[0]["text"]

    def test_parse_empty(self):
        from core.parser import _parse_conversation
        msgs = _parse_conversation("", conv_id=1)
        assert msgs == []

    def test_msg_id_assigned(self):
        msgs = list(FAKE_MESSAGES)
        for i, msg in enumerate(msgs, start=1):
            assert msg["msg_id"] == i


# ── Embedder tests ────────────────────────────────────────────────────────────

class TestEmbedder:
    def test_single_embed_shape(self):
        from core.embedder import embed
        v = embed("Hello world")
        assert v.shape == (1, 384)
        assert v.dtype == np.float32

    def test_batch_embed_shape(self):
        from core.embedder import embed
        texts = ["Hello", "World", "Test"]
        v = embed(texts)
        assert v.shape == (3, 384)

    def test_similar_texts_higher_sim(self):
        from core.embedder import embed, cosine_similarity
        v_hiking1 = embed("I love hiking in the mountains")[0]
        v_hiking2 = embed("Trekking outdoors is my favourite activity")[0]
        v_cooking = embed("I enjoy making pasta and Italian food")[0]
        sim_related   = cosine_similarity(v_hiking1, v_hiking2)
        sim_unrelated = cosine_similarity(v_hiking1, v_cooking)
        assert sim_related > sim_unrelated, \
            f"Expected {sim_related:.3f} > {sim_unrelated:.3f}"

    def test_cosine_sim_range(self):
        from core.embedder import embed, cosine_similarity
        v1 = embed("abc")[0]
        v2 = embed("xyz")[0]
        sim = cosine_similarity(v1, v2)
        assert -1.0 <= sim <= 1.0


# ── Topic detector tests ──────────────────────────────────────────────────────

class TestTopicDetector:
    def test_boundaries_always_start_at_zero(self):
        from core.topic_detector import detect_topic_boundaries
        boundaries = detect_topic_boundaries(FAKE_MESSAGES)
        assert boundaries[0] == 0

    def test_boundaries_are_sorted(self):
        from core.topic_detector import detect_topic_boundaries
        boundaries = detect_topic_boundaries(FAKE_MESSAGES)
        assert boundaries == sorted(boundaries)

    def test_build_segments_coverage(self):
        from core.topic_detector import detect_topic_boundaries, build_topic_segments
        boundaries = detect_topic_boundaries(FAKE_MESSAGES)
        segs = build_topic_segments(FAKE_MESSAGES, boundaries)
        # Every message should appear in exactly one segment
        covered_ids = set()
        for seg in segs:
            for msg in seg["messages"]:
                assert msg["msg_id"] not in covered_ids, "Duplicate message in segments"
                covered_ids.add(msg["msg_id"])
        all_ids = {m["msg_id"] for m in FAKE_MESSAGES}
        assert covered_ids == all_ids, "Not all messages covered by segments"

    def test_segment_fields(self):
        from core.topic_detector import detect_topic_boundaries, build_topic_segments
        boundaries = [0]
        segs = build_topic_segments(FAKE_MESSAGES, boundaries)
        assert segs[0]["topic_id"] == 1
        assert segs[0]["start_msg_id"] == FAKE_MESSAGES[0]["msg_id"]
        assert segs[0]["end_msg_id"] == FAKE_MESSAGES[-1]["msg_id"]
        assert segs[0]["summary"] is None


# ── Persona merger tests ──────────────────────────────────────────────────────

class TestPersonaMerger:
    def test_dedup_habits(self):
        from core.persona_extractor import _merge_personas
        partials = [
            {"habits": ["late sleeper", "Drinks coffee"], "personal_facts": [], "personality_traits": [], "communication_style": {"message_length": "short", "tone": "casual", "emoji_usage": "none", "notable_patterns": []}},
            {"habits": ["Late Sleeper", "exercises daily"], "personal_facts": [], "personality_traits": [], "communication_style": {"message_length": "short", "tone": "casual", "emoji_usage": "none", "notable_patterns": []}},
        ]
        merged = _merge_personas(partials)
        habits_lower = [h.lower() for h in merged["habits"]]
        assert habits_lower.count("late sleeper") == 1, "Duplicate habit not removed"

    def test_mode_voting(self):
        from core.persona_extractor import _merge_personas
        partials = [
            {"habits": [], "personal_facts": [], "personality_traits": [], "communication_style": {"message_length": "short", "tone": "casual", "emoji_usage": "none", "notable_patterns": []}},
            {"habits": [], "personal_facts": [], "personality_traits": [], "communication_style": {"message_length": "short", "tone": "formal", "emoji_usage": "none", "notable_patterns": []}},
            {"habits": [], "personal_facts": [], "personality_traits": [], "communication_style": {"message_length": "long",  "tone": "casual", "emoji_usage": "none", "notable_patterns": []}},
        ]
        merged = _merge_personas(partials)
        assert merged["communication_style"]["tone"] == "casual"       # 2 vs 1
        assert merged["communication_style"]["message_length"] == "short"  # 2 vs 1

    def test_empty_partials(self):
        from core.persona_extractor import _merge_personas
        merged = _merge_personas([{}, {}])
        assert "habits" in merged
        assert isinstance(merged["habits"], list)


# ── Retriever tests (requires built index) ────────────────────────────────────

class TestRetrieverIfIndexExists:
    """These tests run only if the FAISS index has been built."""

    @pytest.fixture(autouse=True)
    def skip_if_no_index(self):
        from config import FAISS_INDEX_PATH
        if not FAISS_INDEX_PATH.exists():
            pytest.skip("FAISS index not built — run build_pipeline.py first")

    def test_retrieve_returns_dict(self):
        from core.retriever import retrieve
        result = retrieve("hobbies and interests")
        assert "topic_summaries" in result
        assert "chunks" in result
        assert "combined_context" in result

    def test_chunks_have_scores(self):
        from core.retriever import retrieve
        result = retrieve("outdoor activities")
        for chunk in result["chunks"]:
            assert "score" in chunk
            assert isinstance(chunk["score"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
