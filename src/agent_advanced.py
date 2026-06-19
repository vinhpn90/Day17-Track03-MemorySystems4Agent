from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model, invoke_with_retry


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""
        use_live = not self.force_offline and self.config.model.api_key
        
        if use_live:
            # Update and persist profile using decay and conflict resolution (Step 9 Bonus)
            serialized = self._update_and_persist_profile(user_id, thread_id, message)
            
            # Append message to memory manager
            self.compact_memory.append(thread_id, "user", message)
            
            # Estimate prompt context tokens
            prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
            
            # Construct live prompt context for the LLM
            ctx = self.compact_memory.context(thread_id)
            summary = ctx.get("summary", "")
            recent_msgs = ctx.get("messages", [])
            
            system_prompt = (
                "Bạn là một trợ lý AI hữu ích.\n"
                "Thông tin lưu trữ về người dùng (User Profile):\n"
                f"{serialized}\n\n"
                "Tóm tắt các cuộc hội thoại trước đó (Summary):\n"
                f"{summary}\n\n"
                "Hãy trả lời tin nhắn mới của người dùng dưới đây, sử dụng thông tin lưu trữ nếu họ hỏi recall."
            )
            
            chat_model = build_chat_model(self.config.model)
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            
            langchain_msgs = [SystemMessage(content=system_prompt)]
            for msg in recent_msgs:
                if msg["role"] == "user":
                    langchain_msgs.append(HumanMessage(content=msg["content"]))
                else:
                    langchain_msgs.append(AIMessage(content=msg["content"]))
                    
            response = invoke_with_retry(chat_model, langchain_msgs)
            reply_text = response.content
            if isinstance(reply_text, list):
                reply_text = "".join(part["text"] if isinstance(part, dict) and "text" in part else str(part) for part in reply_text)
            
            # Append assistant reply
            self.compact_memory.append(thread_id, "assistant", reply_text)
            
            output_tokens = estimate_tokens(reply_text)
            
            self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + output_tokens
            self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
            
            return {
                "response": reply_text,
                "token_usage": output_tokens,
                "prompt_tokens_processed": prompt_tokens
            }
        else:
            return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _update_and_persist_profile(self, user_id: str, thread_id: str, message: str) -> str:
        """Step 9 Bonus: Update profile, handle conflicts, and decay old session facts."""
        # 1. Check for cross-session decay
        is_new_thread = False
        if not hasattr(self, "last_user_threads"):
            self.last_user_threads = {}
        if user_id in self.last_user_threads and self.last_user_threads[user_id] != thread_id:
            is_new_thread = True
        self.last_user_threads[user_id] = thread_id

        # 2. Extract updates (applies confidence filtering internally)
        updates = extract_profile_updates(message)

        # 3. Parse existing facts and their strengths
        current_text = self.profile_store.read_text(user_id)
        facts = {}
        strengths = {}
        
        if current_text:
            for line in current_text.splitlines():
                if line.startswith("- ") and ":" in line:
                    strength = 5
                    if " | strength: " in line:
                        parts = line[2:].split(" | strength: ", 1)
                        kv = parts[0]
                        strength = int(parts[1])
                    else:
                        kv = line[2:]
                    
                    if ":" in kv:
                        k, v = kv.split(":", 1)
                        facts[k.strip()] = v.strip()
                        strengths[k.strip()] = strength

        # 4. Apply decay if it is a new session/thread (unmentioned facts lose strength)
        if is_new_thread:
            decayed_keys = []
            for k in list(facts.keys()):
                # If key is NOT in current turn's updates, decrement strength
                if k not in updates:
                    strengths[k] = strengths[k] - 1
                    if strengths[k] <= 0:
                        decayed_keys.append(k)
            for k in decayed_keys:
                facts.pop(k)
                strengths.pop(k)

        # 5. Overwrite/merge new facts (Conflict Resolution) and reset strength to max
        for k, v in updates.items():
            facts[k] = v
            strengths[k] = 5

        # 6. Serialize and persist
        serialized = "\n".join(f"- {k}: {v} | strength: {strengths[k]}" for k, v in facts.items())
        self.profile_store.write_text(user_id, serialized)
        return serialized

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement the deterministic advanced path."""
        # 1 & 2. Update and persist profile using decay and conflict resolution (Step 9 Bonus)
        self._update_and_persist_profile(user_id, thread_id, message)
        
        # 3. Append to memory manager
        self.compact_memory.append(thread_id, "user", message)
        
        # 4. Estimate prompt-context load
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        
        # 5. Generate reply
        response_text = self._offline_response(user_id, thread_id, message)
        
        # 6. Append assistant reply
        self.compact_memory.append(thread_id, "assistant", response_text)
        
        output_tokens = estimate_tokens(response_text)
        
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + output_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        
        return {
            "response": response_text,
            "token_usage": output_tokens,
            "prompt_tokens_processed": prompt_tokens
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Student TODO: estimate the context carried into one turn."""
        profile_content = self.profile_store.read_text(user_id)
        profile_tokens = estimate_tokens(profile_content)
        
        ctx = self.compact_memory.context(thread_id)
        summary_tokens = estimate_tokens(ctx.get("summary", ""))
        
        messages_tokens = sum(estimate_tokens(msg["content"]) for msg in ctx.get("messages", []))
        
        return profile_tokens + summary_tokens + messages_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Student TODO: return a deterministic answer using persisted memory."""
        current_text = self.profile_store.read_text(user_id)
        if not current_text:
            return "Tôi chưa có thông tin gì về bạn."
        
        facts = {}
        for line in current_text.splitlines():
            if line.startswith("- ") and ":" in line:
                # Handle strength suffix if present
                if " | strength: " in line:
                    kv = line[2:].split(" | strength: ", 1)[0]
                else:
                    kv = line[2:]
                
                if ":" in kv:
                    parts = kv.split(":", 1)
                    facts[parts[0].strip()] = parts[1].strip()
        
        parts = []
        if "Tên" in facts:
            parts.append(f"Tên bạn là {facts['Tên']}.")
        if "Nơi ở" in facts:
            parts.append(f"Bạn hiện ở {facts['Nơi ở']}.")
        if "Nghề nghiệp" in facts:
            parts.append(f"Nghề nghiệp của bạn là {facts['Nghề nghiệp']}.")
        if "Đồ uống" in facts:
            parts.append(f"Đồ uống yêu thích là {facts['Đồ uống']}.")
        if "Món ăn" in facts:
            parts.append(f"Món ăn yêu thích là {facts['Món ăn']}.")
        if "Thú cưng" in facts:
            parts.append(f"Bạn nuôi {facts['Thú cưng']}.")
        if "Style" in facts:
            parts.append(f"Bạn thích style trả lời {facts['Style']}.")
        if "Quan tâm" in facts:
            parts.append(f"Bạn quan tâm đến {facts['Quan tâm']}.")
            
        return " ".join(parts)

    def _maybe_build_langchain_agent(self):
        """Student TODO: wire a live agent with tools and compact middleware."""
        if not self.force_offline and self.config.model.api_key:
            self.langchain_agent = build_chat_model(self.config.model)
        return self.langchain_agent
