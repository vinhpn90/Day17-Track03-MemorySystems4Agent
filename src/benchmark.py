from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Student TODO: read JSON conversations from disk."""
    import json
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Student TODO: return 0 / 0.5 / 1 depending on how many expected facts appear."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = 0
    for fact in expected:
        if fact.lower() in ans_lower:
            matches += 1
            
    val = matches / len(expected)
    if val == 1.0:
        return 1.0
    elif val >= 0.5:
        return 0.5
    else:
        return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Student TODO: add a lightweight quality score for offline mode."""
    if not answer or len(answer.strip()) < 10:
        return 0.0
    
    score = 0.5  # Base score
    # Check if answer contains bullet points
    if "-" in answer or "*" in answer or any(f"{i}." in answer for i in range(1, 5)):
        score += 0.3
    # Check readability (length)
    if len(answer) > 40:
        score += 0.2
        
    return min(1.0, score)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Student TODO: evaluate one agent over many conversations."""
    total_tokens_only = 0
    total_prompt_processed = 0
    recall_scores = []
    quality_scores = []
    
    start_sizes = {}
    end_sizes = {}
    total_compactions = 0
    
    # Initialize start sizes and clean up any existing profiles
    for conv in conversations:
        user_id = conv.get("user_id", "default_user")
        if hasattr(agent, "profile_store"):
            p_path = agent.profile_store.path_for(user_id)
            if p_path.exists():
                p_path.unlink()
            start_sizes[user_id] = 0
            
    for idx, conv in enumerate(conversations):
        print(f"  [{agent_name}] Running conversation {idx+1}/{len(conversations)} (ID: {conv.get('id')})...", flush=True)
        user_id = conv.get("user_id", "default_user")
        conv_id = conv.get("id", "default_conv")
        thread_id = conv_id
        
        # Feed all turns
        for turn_idx, turn in enumerate(conv.get("turns", [])):
            print(f"    - Turn {turn_idx+1}/{len(conv.get('turns', []))}", flush=True)
            res = agent.reply(user_id, thread_id, turn)
            
        # Accumulate metrics
        total_tokens_only += agent.token_usage(thread_id)
        total_prompt_processed += agent.prompt_token_usage(thread_id)
        total_compactions += agent.compaction_count(thread_id)
        
        # Evaluate recall & quality in a fresh thread
        recall_thread_id = f"recall-{conv_id}"
        for rq_idx, rq in enumerate(conv.get("recall_questions", [])):
            question = rq.get("question", "")
            expected = rq.get("expected_contains", [])
            print(f"    - Recall Query {rq_idx+1}/{len(conv.get('recall_questions', []))}: {question[:30]}...", flush=True)
            res = agent.reply(user_id, recall_thread_id, question)
            ans = res.get("response", "")
            
            # Accumulate query tokens
            total_tokens_only += res.get("token_usage", 0)
            total_prompt_processed += res.get("prompt_tokens_processed", 0)
            
            r_score = recall_points(ans, expected)
            q_score = heuristic_quality(ans, expected)
            
            recall_scores.append(r_score)
            quality_scores.append(q_score)
            
        if hasattr(agent, "profile_store"):
            end_sizes[user_id] = agent.profile_store.file_size(user_id)
            
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    memory_growth = 0
    if hasattr(agent, "profile_store"):
        for user_id in end_sizes:
            memory_growth += (end_sizes[user_id] - start_sizes.get(user_id, 0))
            
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_tokens_only,
        prompt_tokens_processed=total_prompt_processed,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Student TODO: print a markdown table or tabulated output."""
    from tabulate import tabulate
    table_data = []
    for row in rows:
        table_data.append([
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.2f}",
            f"{row.response_quality:.2f}",
            row.memory_growth_bytes,
            row.compactions
        ])
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions"
    ]
    return tabulate(table_data, headers=headers, tablefmt="github")


def make_html_table(rows: list[BenchmarkRow]) -> str:
    html = "<table>\n<thead>\n<tr>"
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions"
    ]
    for h in headers:
        html += f"<th>{h}</th>"
    html += "</tr>\n</thead>\n<tbody>\n"
    for r in rows:
        html += "<tr>"
        html += f"<td><strong>{r.agent_name}</strong></td>"
        html += f"<td>{r.agent_tokens_only}</td>"
        html += f"<td>{r.prompt_tokens_processed}</td>"
        html += f"<td>{r.recall_score:.2f}</td>"
        html += f"<td>{r.response_quality:.2f}</td>"
        html += f"<td>{r.memory_growth_bytes}</td>"
        html += f"<td>{r.compactions}</td>"
        html += "</tr>\n"
    html += "</tbody>\n</table>"
    return html


def main() -> None:
    """Student TODO: run both benchmark suites."""
    config = load_config(Path(__file__).resolve().parent.parent)
    
    std_conv_path = config.data_dir / "conversations.json"
    stress_conv_path = config.data_dir / "advanced_long_context.json"
    
    std_conversations = load_conversations(std_conv_path)
    stress_conversations = load_conversations(stress_conv_path)
    
    # Determine force_offline dynamically based on API key presence
    force_offline = not bool(config.model.api_key)
    print(f"Running benchmark in {'OFFLINE' if force_offline else 'LIVE'} mode using model: {config.model.model_name}")
    
    print("--- Running Standard Benchmark ---")
    baseline_agent = BaselineAgent(config, force_offline=force_offline)
    advanced_agent = AdvancedAgent(config, force_offline=force_offline)
    
    baseline_std_row = run_agent_benchmark("Baseline Agent", baseline_agent, std_conversations, config)
    advanced_std_row = run_agent_benchmark("Advanced Agent", advanced_agent, std_conversations, config)
    
    std_output = format_rows([baseline_std_row, advanced_std_row])
    print(std_output)
    print()
    
    print("--- Running Long-Context Stress Benchmark ---")
    baseline_agent_stress = BaselineAgent(config, force_offline=force_offline)
    advanced_agent_stress = AdvancedAgent(config, force_offline=force_offline)
    
    baseline_stress_row = run_agent_benchmark("Baseline Agent", baseline_agent_stress, stress_conversations, config)
    advanced_stress_row = run_agent_benchmark("Advanced Agent", advanced_agent_stress, stress_conversations, config)
    
    stress_output = format_rows([baseline_stress_row, advanced_stress_row])
    print(stress_output)
    print()

    # Generate Markdown Report
    md_content = f"""# Memory Systems Benchmark Report

- **Model Name:** `{config.model.model_name}`
- **Mode:** `{'OFFLINE' if force_offline else 'LIVE'}`

## Standard Benchmark
{std_output}

## Long-Context Stress Benchmark
{stress_output}

## Phân tích kết quả & Đánh giá Trade-off

### 1. Phân tích hiệu quả của Compact Memory
- **Hội thoại ngắn:** Trong các hội thoại ngắn (như Standard Benchmark), cơ chế Compact Memory không mang lại nhiều lợi thế về mặt token, thậm chí có thể tốn thêm token do cần lưu trữ và xử lý phần `Summary` (tóm tắt) và `User Profile` làm tăng độ dài prompt đầu vào.
- **Hội thoại dài (Stress Benchmark):** Khi số lượng lượt hội thoại tăng lên rất nhiều, Baseline Agent phải nhét toàn bộ lịch sử trò chuyện vào prompt, làm cho số lượng `Prompt tokens processed` tăng vọt theo cấp số nhân. Trong khi đó, Advanced Agent sử dụng Compact Memory để nén các tin nhắn cũ thành Summary và giữ lại số tin nhắn gần nhất. Điều này giúp giữ cho kích thước prompt ổn định và **tối ưu lượng prompt token processed** cực kỳ hiệu quả khi mạch hội thoại kéo dài.

### 2. Phân tích rủi ro & Giải pháp giảm thiểu (Bonus Features)
- **Rủi ro Memory Bloat (Phình to bộ nhớ):** Nếu lưu giữ tất cả các thông tin vĩnh viễn, file `User.md` sẽ phình to không giới hạn.
  - *Giải pháp:* Cơ chế **Memory Decay** được tích hợp. Các thông tin không được nhắc lại trong các session mới sẽ bị giảm độ bền vững (strength) từ 5 xuống 0 và tự động xóa bỏ khi về 0.
- **Rủi ro lưu sai Fact (Noise & Joke):** Người dùng có thể đùa hoặc nói các thông tin không chính xác về bản thân.
  - *Giải pháp:* Bộ lọc **Confidence Threshold** lọc bỏ các câu hỏi hoặc các câu đùa (ví dụ: đùa chuyển sang làm product manager) để tránh ghi nhận sai thông tin.
- **Rủi ro mâu thuẫn thông tin (Conflict):** Người dùng đính chính hoặc cập nhật thông tin mới (ví dụ chuyển nơi ở từ Huế sang Đà Nẵng).
  - *Giải pháp:* Cơ chế **Conflict Handling** tự động phát hiện key trùng lặp và ghi đè thông tin mới nhất lên thông tin cũ, giúp dữ liệu luôn nhất quán.
"""
    report_md_path = config.base_dir / "benchmark_report.md"
    report_md_path.write_text(md_content, encoding="utf-8")
    print(f"Saved Markdown report to: {report_md_path}")

    # Generate HTML Report
    mode_text = 'OFFLINE' if force_offline else 'LIVE'
    badge_class = 'badge-offline' if force_offline else 'badge-live'
    std_table_html = make_html_table([baseline_std_row, advanced_std_row])
    stress_table_html = make_html_table([baseline_stress_row, advanced_stress_row])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Memory Systems Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0d1117;
            --container-bg: #161b22;
            --text-color: #c9d1d9;
            --heading-color: #f0f6fc;
            --accent-color: #58a6ff;
            --border-color: #30363d;
            --table-header-bg: #21262d;
            --table-row-hover: #1f242c;
        }}
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 900px;
            width: 100%;
            background-color: var(--container-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }}
        h1 {{
            color: var(--heading-color);
            font-size: 2.5rem;
            margin-top: 0;
            font-weight: 800;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 15px;
        }}
        h2 {{
            color: var(--accent-color);
            font-size: 1.5rem;
            margin-top: 30px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .metadata {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            background-color: rgba(88, 166, 255, 0.05);
            padding: 15px;
            border-radius: 8px;
            border: 1px solid rgba(88, 166, 255, 0.15);
        }}
        .metadata-item {{
            font-size: 0.95rem;
        }}
        .metadata-item strong {{
            color: var(--heading-color);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: var(--table-header-bg);
            color: var(--heading-color);
            font-weight: 600;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        td {{
            font-size: 0.95rem;
        }}
        tr:last-child td {{
            border-bottom: none;
        }}
        tr:hover {{
            background-color: var(--table-row-hover);
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .badge-live {{
            background-color: rgba(46, 160, 67, 0.15);
            color: #56d364;
            border: 1px solid rgba(46, 160, 67, 0.4);
        }}
        .badge-offline {{
            background-color: rgba(248, 81, 73, 0.15);
            color: #ff7b72;
            border: 1px solid rgba(248, 81, 73, 0.4);
        }}
        .analysis {{
            margin-top: 40px;
            border-top: 1px solid var(--border-color);
            padding-top: 30px;
        }}
        .analysis h3 {{
            color: var(--heading-color);
            font-size: 1.2rem;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .analysis p {{
            line-height: 1.6;
            margin-bottom: 15px;
            font-size: 0.95rem;
        }}
        .analysis ul {{
            margin-bottom: 20px;
            padding-left: 20px;
        }}
        .analysis li {{
            margin-bottom: 8px;
            line-height: 1.5;
            font-size: 0.95rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Memory Systems Benchmark Report</h1>
        <div class="metadata">
            <div class="metadata-item"><strong>Model Name:</strong> {config.model.model_name}</div>
            <div class="metadata-item"><strong>Mode:</strong> <span class="badge {badge_class}">{mode_text}</span></div>
        </div>
        
        <h2>Standard Benchmark</h2>
        {std_table_html}
        
        <h2>Long-Context Stress Benchmark</h2>
        {stress_table_html}

        <div class="analysis">
            <h2>Phân tích kết quả & Đánh giá Trade-off</h2>
            
            <h3>1. Phân tích hiệu quả của Compact Memory</h3>
            <p>
                <strong>Hội thoại ngắn:</strong> Trong các hội thoại ngắn (như Standard Benchmark), cơ chế Compact Memory không mang lại nhiều lợi thế về mặt token, thậm chí có thể tốn thêm token do cần lưu trữ và xử lý phần <em>Summary</em> (tóm tắt) và <em>User Profile</em> làm tăng độ dài prompt đầu vào.
            </p>
            <p>
                <strong>Hội thoại dài (Stress Benchmark):</strong> Khi số lượng lượt hội thoại tăng lên rất nhiều, Baseline Agent phải nhét toàn bộ lịch sử trò chuyện vào prompt, làm cho số lượng <em>Prompt tokens processed</em> tăng vọt theo cấp số nhân. Trong khi đó, Advanced Agent sử dụng Compact Memory để nén các tin nhắn cũ thành Summary và giữ lại số tin nhắn gần nhất. Điều này giúp giữ cho kích thước prompt ổn định và <strong>tối ưu lượng prompt token processed</strong> cực kỳ hiệu quả khi mạch hội thoại kéo dài.
            </p>

            <h3>2. Phân tích rủi ro & Giải pháp giảm thiểu (Bonus Features)</h3>
            <ul>
                <li><strong>Rủi ro Memory Bloat (Phình to bộ nhớ):</strong> Nếu lưu giữ tất cả các thông tin vĩnh viễn, file <em>User.md</em> sẽ phình to không giới hạn.
                    <br><em>Giải pháp:</em> Cơ chế <strong>Memory Decay</strong> được tích hợp. Các thông tin không được nhắc lại trong các session mới sẽ bị giảm độ bền vững (strength) từ 5 xuống 0 và tự động xóa bỏ khi về 0.
                </li>
                <li><strong>Rủi ro lưu sai Fact (Noise & Joke):</strong> Người dùng có thể đùa hoặc nói các thông tin không chính xác về bản thân.
                    <br><em>Giải pháp:</em> Bộ lọc <strong>Confidence Threshold</strong> lọc bỏ các câu hỏi hoặc các câu đùa (ví dụ: đùa chuyển sang làm product manager) để tránh ghi nhận sai thông tin.
                </li>
                <li><strong>Rủi ro mâu thuẫn thông tin (Conflict):</strong> Người dùng đính chính hoặc cập nhật thông tin mới (ví dụ chuyển nơi ở từ Huế sang Đà Nẵng).
                    <br><em>Giải pháp:</em> Cơ chế <strong>Conflict Handling</strong> tự động phát hiện key trùng lặp và ghi đè thông tin mới nhất lên thông tin cũ, giúp dữ liệu luôn nhất quán.
                </li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
    report_html_path = config.base_dir / "benchmark_report.html"
    report_html_path.write_text(html_content, encoding="utf-8")
    print(f"Saved HTML report to: {report_html_path}")


if __name__ == "__main__":
    main()
