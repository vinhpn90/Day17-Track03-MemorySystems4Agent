from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def make_config(tmp_path: Path):
    """Student TODO: build an isolated config for tests."""
    from config import LabConfig
    from model_provider import ProviderConfig
    
    provider_cfg = ProviderConfig(
        provider="gemini",
        model_name="gemini-1.5-flash",
        temperature=0.0
    )
    
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path,
        state_dir=tmp_path,
        compact_threshold_tokens=50,  # low threshold to trigger easily in tests
        compact_keep_messages=2,     # keep only 2 messages
        model=provider_cfg,
        judge_model=provider_cfg
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Student TODO: verify `User.md` can be created, updated, and edited."""
    from memory_store import UserProfileStore
    
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user_123"
    
    # Write initial profile
    content = "- Tên: John Doe\n- Nơi ở: New York"
    store.write_text(user_id, content)
    
    # Read and verify
    assert store.read_text(user_id) == content
    
    # Edit profile
    success = store.edit_text(user_id, "New York", "Chicago")
    assert success is True
    assert "- Nơi ở: Chicago" in store.read_text(user_id)
    
    # File size check
    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Student TODO: verify long threads trigger compaction."""
    from memory_store import CompactMemoryManager
    
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2)
    thread_id = "test_thread"
    
    # Message 1 (approx 30 tokens)
    manager.append(thread_id, "user", "A" * 120)
    assert manager.compaction_count(thread_id) == 0
    
    # Message 2 (approx 30 tokens) -> Total 60 > 50, but messages len = 2.
    # Compaction requires > keep_messages (2) to actually have messages to summarize.
    manager.append(thread_id, "assistant", "B" * 120)
    assert manager.compaction_count(thread_id) == 0
    
    # Message 3 (approx 30 tokens) -> Total 90 > 50, messages len = 3 > 2. Triggers compaction.
    manager.append(thread_id, "user", "C" * 120)
    assert manager.compaction_count(thread_id) > 0
    assert len(manager.context(thread_id)["messages"]) == 2
    assert manager.context(thread_id)["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Student TODO: verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)
    
    user_id = "recall_test_user"
    
    # Thread 1: User provides name
    baseline.reply(user_id, "thread_1", "Chào bạn, mình tên là DũngCT.")
    advanced.reply(user_id, "thread_1", "Chào bạn, mình tên là DũngCT.")
    
    # Thread 2: Fresh thread. Ask for the name
    baseline_res = baseline.reply(user_id, "thread_2", "Mình tên gì?")
    advanced_res = advanced.reply(user_id, "thread_2", "Mình tên gì?")
    
    # Baseline has no User.md, so it shouldn't remember
    assert "DũngCT" not in baseline_res["response"]
    
    # Advanced uses User.md, so it should remember
    assert "DũngCT" in advanced_res["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Student TODO: compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    
    baseline = BaselineAgent(cfg, force_offline=True)
    advanced = AdvancedAgent(cfg, force_offline=True)
    
    user_id = "token_test_user"
    thread_id = "long_thread"
    
    # Feed multiple turns of ~15 tokens each
    for i in range(8):
        baseline.reply(user_id, thread_id, "Hello " * 15)
        advanced.reply(user_id, thread_id, "Hello " * 15)
        
    baseline_prompt_tokens = baseline.prompt_token_usage(thread_id)
    advanced_prompt_tokens = advanced.prompt_token_usage(thread_id)
    
    # Advanced should process fewer prompt tokens because of compaction
    assert advanced_prompt_tokens < baseline_prompt_tokens


def test_bonus_features(tmp_path: Path) -> None:
    """Verify Conflict Handling, Confidence Threshold, and Memory Decay."""
    cfg = make_config(tmp_path)
    advanced = AdvancedAgent(cfg, force_offline=True)
    user_id = "bonus_test_user"

    # 1. Verify Confidence Threshold: joke statement about product manager should not be saved
    advanced.reply(user_id, "thread_1", "Mình đùa thôi, mình chuyển sang làm product manager rồi.")
    profile_text = advanced.profile_store.read_text(user_id)
    assert "product manager" not in profile_text

    # 2. Save valid facts
    advanced.reply(user_id, "thread_1", "Mình tên là DũngCT.")
    advanced.reply(user_id, "thread_1", "Mình ở Đà Nẵng.")
    profile_text = advanced.profile_store.read_text(user_id)
    assert "DũngCT" in profile_text
    assert "Đà Nẵng" in profile_text
    assert "strength: 5" in profile_text

    # 3. Verify Conflict Handling: update location from Đà Nẵng to Huế
    advanced.reply(user_id, "thread_1", "Giờ mình chuyển sang ở Huế rồi.")
    profile_text = advanced.profile_store.read_text(user_id)
    assert "Huế" in profile_text
    assert "Đà Nẵng" not in profile_text  # old location must be overwritten

    # 4. Verify Memory Decay: transition across threads decays unmentioned facts
    # Thread 2: Start a new session. Do not mention "Tên" or "Nơi ở"
    advanced.reply(user_id, "thread_2", "Mình thích uống cà phê sữa đá.")
    profile_text = advanced.profile_store.read_text(user_id)
    assert "cà phê sữa đá" in profile_text
    assert "strength: 5" in profile_text
    assert "Tên: DũngCT | strength: 4" in profile_text
    assert "Nơi ở: Huế | strength: 4" in profile_text

    # Decay facts until strength becomes 0 (requires 4 more threads)
    for t_idx in range(3, 7):
        advanced.reply(user_id, f"thread_{t_idx}", "Một câu nói linh tinh.")
    
    # After thread_6, the facts should be decayed and removed completely
    profile_text = advanced.profile_store.read_text(user_id)
    assert "DũngCT" not in profile_text
    assert "Huế" not in profile_text
    assert "cà phê sữa đá" in profile_text
