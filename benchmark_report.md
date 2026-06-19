# Memory Systems Benchmark Report

- **Model Name:** `/var/lib/vllm/hf/gpt-oss-120b`
- **Mode:** `LIVE`

## Standard Benchmark
| Agent          |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline Agent |               85620 |                    395041 |                   0.11 |               0.96 |                       0 |             0 |
| Advanced Agent |               28815 |                     66755 |                   0.68 |               0.93 |                      85 |           116 |

## Long-Context Stress Benchmark
| Agent          |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline Agent |                5482 |                     59854 |                   0    |                0.9 |                       0 |             0 |
| Advanced Agent |                5985 |                     18167 |                   0.67 |                1   |                     185 |            28 |

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
