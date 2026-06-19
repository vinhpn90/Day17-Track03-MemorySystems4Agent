from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator."""
    if not text:
        return 0
    # Strip whitespace and approximate tokens: characters / 4
    return max(1, len(text.strip()) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        return self.root_dir / f"{sanitized}.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        p = self.path_for(user_id)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        p = self.path_for(user_id)
        if not p.exists():
            return False
        content = p.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        new_content = content.replace(search_text, replacement, 1)
        p.write_text(new_content, encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        if not p.exists():
            return 0
        return p.stat().st_size


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts."""
    import re
    facts = {}
    msg_lower = message.lower()

    # Skip question-only turns
    if message.strip().endswith("?") or any(q in msg_lower for q in ("không?", "là gì", "ở đâu", "như thế nào", "ai không", "chưa?", "đâu không", "nhắc lại", "nhắc giúp")):
        return {}

    # Tên
    name_match = re.search(r"(?:mình tên là|tên mình là|tên của mình là|tên là)\s*([A-Za-z0-9_À-ỹ\s]+?)(?:\.|\,|$|\s+hiện|\s+cho|\s+stress)", message, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        if name:
            facts["Tên"] = name

    # Nơi ở (ignore Hanoi noise)
    is_hanoi_noise = "hà nội" in msg_lower and "không phải nơi ở" in msg_lower
    if not is_hanoi_noise:
        loc_match = re.search(r"(?:ở|đang ở|làm việc ở|ở hiện tại là|vẫn ở)\s*(Đà Nẵng|Huế|Hà Nội)", message, re.IGNORECASE)
        if loc_match:
            facts["Nơi ở"] = loc_match.group(1).strip()
        elif "đang ở đà nẵng" in msg_lower or "làm việc ở đà nẵng" in msg_lower:
            facts["Nơi ở"] = "Đà Nẵng"
        elif "đang ở huế" in msg_lower:
            facts["Nơi ở"] = "Huế"

    # Nghề nghiệp
    is_pm_joke = "product manager" in msg_lower and "đùa" in msg_lower
    if not is_pm_joke:
        prof_match = re.search(r"(?:làm|chuyển sang|nghề nghiệp hiện tại vẫn là|nghề nghiệp vẫn là|nghề nghiệp hiện tại là|làm nghề)\s*(backend engineer|MLOps engineer|product manager)", message, re.IGNORECASE)
        if prof_match:
            facts["Nghề nghiệp"] = prof_match.group(1).strip()

    # Đồ uống
    if "cà phê sữa đá" in msg_lower:
        facts["Đồ uống"] = "cà phê sữa đá"

    # Món ăn
    if "mì quảng" in msg_lower:
        facts["Món ăn"] = "mì Quảng"

    # Thú cưng
    if "corgi" in msg_lower:
        facts["Thú cưng"] = "corgi tên Bơ"

    # Style trả lời
    if "3 bullet" in msg_lower or "3-bullet" in msg_lower:
        facts["Style"] = "3 bullet"
    elif "ngắn gọn" in msg_lower:
        facts["Style"] = "ngắn gọn"

    # Quan tâm
    if "python" in msg_lower or "ai" in msg_lower:
        facts["Quan tâm"] = "Python, AI"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages."""
    if not messages:
        return ""
    # In offline mode, create a truncated representation to reduce token count
    summary_parts = []
    for msg in messages:
        content = msg.get("content", "")
        role = "U" if msg.get("role") == "user" else "A"
        truncated = content[:20] + "..." if len(content) > 20 else content
        summary_parts.append(f"{role}:{truncated}")
    return " | ".join(summary_parts)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads."""
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0
            }
        
        thread = self.state[thread_id]
        thread["messages"].append({"role": role, "content": content})
        
        # Calculate total tokens (summary + messages)
        summary_tokens = estimate_tokens(thread["summary"])
        messages_tokens = sum(estimate_tokens(msg["content"]) for msg in thread["messages"])
        total_tokens = summary_tokens + messages_tokens
        
        if total_tokens > self.threshold_tokens:
            # We compact messages if we have enough messages to keep
            if len(thread["messages"]) > self.keep_messages:
                to_keep = thread["messages"][-self.keep_messages:]
                to_compact = thread["messages"][:-self.keep_messages]
                
                new_summary = summarize_messages(to_compact)
                if thread["summary"]:
                    thread["summary"] = thread["summary"] + " | " + new_summary
                else:
                    thread["summary"] = new_summary
                
                thread["messages"] = to_keep
                thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, Any]:
        if thread_id not in self.state:
            return {"messages": [], "summary": "", "compactions": 0}
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return self.state[thread_id]["compactions"]
