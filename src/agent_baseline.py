from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model, invoke_with_retry


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: return the agent response and token accounting."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        
        use_live = not self.force_offline and self.config.model.api_key
        if use_live:
            session = self.sessions[thread_id]
            session.messages.append({"role": "user", "content": message})
            
            chat_model = build_chat_model(self.config.model)
            from langchain_core.messages import HumanMessage, AIMessage
            
            langchain_msgs = []
            for msg in session.messages:
                if msg["role"] == "user":
                    langchain_msgs.append(HumanMessage(content=msg["content"]))
                else:
                    langchain_msgs.append(AIMessage(content=msg["content"]))
            
            response = invoke_with_retry(chat_model, langchain_msgs)
            reply_text = response.content
            if isinstance(reply_text, list):
                reply_text = "".join(part["text"] if isinstance(part, dict) and "text" in part else str(part) for part in reply_text)
            
            session.messages.append({"role": "assistant", "content": reply_text})
            
            in_tokens = sum(estimate_tokens(msg["content"]) for msg in session.messages[:-1])
            out_tokens = estimate_tokens(reply_text)
            
            session.token_usage += out_tokens
            session.prompt_tokens_processed += in_tokens
            
            return {
                "response": reply_text,
                "token_usage": out_tokens,
                "prompt_tokens_processed": in_tokens
            }
        else:
            return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement a simple offline behavior."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        session = self.sessions[thread_id]
        
        session.messages.append({"role": "user", "content": message})
        
        # Estimate prompt context tokens (all messages in session before the reply)
        prompt_tokens = sum(estimate_tokens(msg["content"]) for msg in session.messages[:-1])
        
        # Standard reply
        response_text = f"Baseline response to: {message[:30]}..."
        
        session.messages.append({"role": "assistant", "content": response_text})
        output_tokens = estimate_tokens(response_text)
        
        session.token_usage += output_tokens
        session.prompt_tokens_processed += prompt_tokens
        
        return {
            "response": response_text,
            "token_usage": output_tokens,
            "prompt_tokens_processed": prompt_tokens
        }

    def _maybe_build_langchain_agent(self):
        """Student TODO: optionally wire `create_agent` + `InMemorySaver` here."""
        if not self.force_offline and self.config.model.api_key:
            self.langchain_agent = build_chat_model(self.config.model)
        return self.langchain_agent
