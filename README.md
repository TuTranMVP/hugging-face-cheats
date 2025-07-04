# 🤖 AI Interview Agent - Hướng dẫn sử dụng

## Tổng quan
AI Interview Agent là một công cụ CLI chatbox tương tác với **tích hợp AI Ollama llama3:8b** để phân tích dữ liệu từ các file .md và mô phỏng phỏng vấn kiến thức về Hugging Face một cách thông minh và chính xác.

## ✨ Tính năng mới - Enhanced Workspace Loading

### 🔍 Workspace Discovery & Smart Loading
- **Auto-discover**: Tự động khám phá cấu trúc workspace
- **Selective Loading**: Chọn folders cụ thể để nạp
- **Multi-format Support**: Markdown, Python, JSON, YAML
- **Intelligent Parsing**: Phân tích code và trích xuất thông tin
- **Performance Optimization**: Cache và tối ưu memory

### 🤖 AI Mode (Ollama llama3:8b) - Enhanced
- **Rich Context**: Xây dựng context từ toàn bộ workspace
- **Smart Relevance**: Tính toán độ liên quan thông minh
- **Knowledge Fusion**: Kết hợp multiple sources
- **Confidence Scoring**: Đánh giá độ tin cậy nâng cao
- **Enhanced Prompting**: System prompts được tối ưu

### 💬 Interactive Commands - Expanded
- **Workspace Stats**: `stats` - Hiển thị thống kê workspace
- **Smart Search**: `search <keyword>` - Tìm kiếm trong knowledge base  
- **Folder Selection**: Tương tác chọn folders khi loading
- **Real-time Metrics**: Theo dõi performance và usage

### 📚 Rule-based Mode (Fallback)
- **Keyword Search**: Tìm kiếm dựa trên từ khóa
- **Smart Extraction**: Trích xuất thông tin cơ bản
- **Always Available**: Hoạt động khi AI không khả dụng
- **Fast Response**: Phản hồi nhanh chóng

### 🔄 Hybrid System
- **Auto-fallback**: Tự động chuyển sang rule-based nếu AI lỗi
- **Toggle Mode**: Chuyển đổi giữa AI và rule-based bằng lệnh `ai`
- **Smart Context**: Xây dựng context thông minh cho AI

## Tính năng chính

### 🎯 Chế độ Interview (Phỏng vấn)
- Phân tích câu hỏi từ file .md
- Tạo phiên phỏng vấn tương tác
- Theo dõi điểm số và thống kê
- Đưa ra lời khuyên dựa trên kết quả

### 💬 Chế độ Chat (Trò chuyện) - **Enhanced với AI**
- **AI-powered Chat**: Trò chuyện thông minh với Ollama LLM
- **Smart Q&A**: Trả lời câu hỏi dựa trên kiến thức đã tải
- **Context Building**: Tự động xây dựng context từ knowledge base
- **Thinking Process**: Hiển thị quá trình suy luận của AI
- **Confidence Scoring**: Đánh giá độ tin cậy câu trả lời

### 📊 Phân tích dữ liệu
- Tự động phân tích file .md
- Trích xuất câu hỏi và kiến thức
- Tạo từ khóa và liên kết ngữ cảnh
- Hiển thị thông tin chi tiết

## Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.11+
- macOS/Linux/Windows
- Terminal/Command Prompt
- **Ollama (Optional - cho AI mode)**

### 2. Cài đặt thư viện
```bash
# Tạo virtual environment
python3 -m venv venv

# Kích hoạt virtual environment
source venv/bin/activate  # macOS/Linux
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 3. Cài đặt Ollama (Optional - cho AI mode)
```bash
# macOS
brew install ollama

# Khởi động Ollama
ollama serve

# Cài đặt model llama3:8b
ollama pull llama3:8b
```

### 4. Cấu trúc thư viện
```
click>=8.0.0           # CLI framework
colorama>=0.4.6        # Terminal colors
rich>=13.0.0           # Rich text formatting
markdown>=3.4.0        # Markdown parsing
beautifulsoup4>=4.12.0 # HTML parsing
openai>=1.0.0          # OpenAI API (optional)
python-dotenv>=1.0.0   # Environment variables
```

## Sử dụng

### Cú pháp cơ bản
```bash
python main.py [FILES...] [OPTIONS]
```

### Các tùy chọn
- `-m, --mode [interview|chat|both]`: Chọn chế độ hoạt động
- `-s, --shuffle`: Xáo trộn thứ tự câu hỏi
- `-l, --limit INTEGER`: Giới hạn số lượng câu hỏi
- `-v, --verbose`: Hiển thị thông tin chi tiết
- `--help`: Hiển thị trợ giúp

### Ví dụ sử dụng nâng cao

#### 1. Workspace Loading (Khuyến nghị)
```bash
# Nạp toàn bộ workspace hiện tại
python main.py --workspace . --mode chat

# Nạp workspace với Python analysis
python main.py --workspace . --include-python --mode both

# Chọn folders cụ thể
python main.py --workspace . --folders "getting-started,pipelines,text-classification"

# Loại trừ folders không cần thiết
python main.py --workspace . --exclude-folders "__pycache__,node_modules" --include-python
```

#### 2. Advanced Configuration
```bash
# Giới hạn kích thước file (2MB)
python main.py --workspace . --max-file-size 2097152

# Auto-discovery mode (không cần chỉ định gì)
python main.py --mode chat

# Verbose mode để debug
python main.py --workspace . --verbose --include-python
```

#### 3. Traditional file loading (vẫn hỗ trợ)
```bash
# File riêng lẻ
python main.py getting-started/questions.md getting-started/introduction.md

# Wildcard patterns
python main.py */questions.md --mode interview

# Với options
python main.py questions.md --shuffle --limit 10 --mode interview
```

## 🤖 Sử dụng AI Mode

### Bước 1: Khởi động Ollama
```bash
# Terminal 1: Khởi động Ollama server
ollama serve

# Terminal 2: Cài đặt model (chỉ cần 1 lần)
ollama pull llama3:8b
```

### Bước 2: Chạy Chat Mode
```bash
python main.py getting-started/introduction.md --mode chat
```

### Bước 3: Sử dụng AI trong Chat
```
💬 Chế độ Chat tương tác (🤖 AI Ollama)
Bạn: Hugging Face là gì?

🤖 AI Assistant
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🤖 AI Ollama Response (Confidence: 85.0%)                                      │
│                                                                                 │
│ 💭 Thinking Process:                                                           │
│ Analyzing the question about Hugging Face, I need to extract key information   │
│ from the knowledge base about this platform...                                 │
│                                                                                 │
│ 📝 Answer:                                                                     │
│ Hugging Face là một nền tảng cộng tác cho cộng đồng AI/ML với các tính năng:  │
│ • Model Repository: Lưu trữ hàng ngàn pre-trained models                       │
│ • Datasets: Bộ sưu tập dữ liệu training                                       │
│ • Spaces: Demo ứng dụng AI tương tác                                           │
│ • Transformers Library: Thư viện Python dễ sử dụng                            │
│                                                                                 │
│ 🔍 Source: AI Analysis + Knowledge Base                                        │
│ 📊 Knowledge Used: 1 documents                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Bước 4: Chuyển đổi chế độ
```
Bạn: ai
System: Chuyển sang chế độ: 📚 Rule-based

Bạn: ai  
System: Chuyển sang chế độ: 🤖 AI Ollama
```

### Các lệnh trong Chat Mode
- `ai` - Chuyển đổi giữa AI và Rule-based
- `interview` - Chuyển sang chế độ phỏng vấn
- `quit` - Thoát chương trình

## 🔧 Demo và Test

### Demo Enhanced Features
```bash
# Chạy demo để xem các tính năng mới
python demo_enhanced_workspace.py

# Test workspace loading
python main.py --workspace . --folders "getting-started" --verbose

# Full workspace với AI (cần Ollama)
ollama serve  # Terminal khác
python main.py --workspace . --include-python --mode chat
```

### Test không cần Ollama
```bash
# Chạy rule-based mode
python main.py getting-started/introduction.md --mode chat
# Hệ thống sẽ tự động fallback sang rule-based
```

## Định dạng file input

### File câu hỏi (.md)
```markdown
## Question 1: exercise_1
**Title:** [Tiêu đề câu hỏi]

**Description:** [Mô tả ngữ cảnh]

**Question:** [Câu hỏi chính]

**Options:**
- [Lựa chọn 1]
- [Lựa chọn 2]
- [Lựa chọn 3]
- [Lựa chọn 4]

**Correct Answer:** [Đáp án đúng]

**Explanation:** [Giải thích]

---
```

### File kiến thức (.md)
```markdown
# Tiêu đề chính

## Phần 1
Nội dung kiến thức...

### Phần con
Chi tiết kiến thức...

## Phần 2
Nội dung khác...
```

## Tương tác trong chương trình

### Chế độ Interview
- Chọn đáp án: `1`, `2`, `3`, `4`
- Trợ giúp: `help` hoặc `h`
- Thoát: `quit` hoặc `q`

### Chế độ Chat
- Đặt câu hỏi: Nhập câu hỏi tự do
- Chuyển sang phỏng vấn: `interview`
- Trợ giúp: `help` hoặc `h`
- Thoát: `quit` hoặc `q`

### Lệnh chung
- `Ctrl+C`: Thoát khẩn cấp
- `Enter`: Xác nhận lựa chọn

## Các tính năng nâng cao

### 1. Phân tích thông minh
- Tự động nhận diện file câu hỏi và kiến thức
- Trích xuất từ khóa quan trọng
- Liên kết ngữ cảnh giữa câu hỏi và kiến thức

### 2. Giao diện người dùng
- Sử dụng Rich library cho giao diện đẹp
- Màu sắc phân loại thông tin
- Progress bar và spinner
- Bảng và panel thông tin

### 3. Thống kê và báo cáo
- Theo dõi điểm số realtime
- Thống kê thời gian làm bài
- Đánh giá và lời khuyên
- Lưu trữ lịch sử (tùy chọn)

### 4. Xử lý lỗi
- Xử lý file không tồn tại
- Validation input người dùng
- Graceful error handling
- Logging chi tiết (verbose mode)

## Khắc phục sự cố

### Lỗi thường gặp

#### 1. ModuleNotFoundError
```
Traceback (most recent call last):
  File "main.py", line 16, in <module>
    import click
ModuleNotFoundError: No module named 'click'
```

**Giải pháp:**
```bash
pip install -r requirements.txt
```

#### 2. File not found
```
❌ Cần chỉ định ít nhất một file .md!
```

**Giải pháp:**
```bash
python main.py getting-started/questions.md getting-started/introduction.md
```

#### 3. Externally managed environment
```
error: externally-managed-environment
```

**Giải pháp:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Debug mode
```bash
python main.py *.md --verbose
```

### Kiểm tra cài đặt
```bash
python main.py --help
```

## Mở rộng

### Thêm câu hỏi mới
1. Mở file questions.md
2. Copy template từ cuối file
3. Thay thế các placeholder
4. Đảm bảo exercise_id duy nhất
5. Test với CLI

### Thêm kiến thức mới
1. Tạo file .md mới
2. Sử dụng cấu trúc heading markdown
3. Thêm từ khóa quan trọng
4. Load với CLI

### Tùy chỉnh giao diện
- Chỉnh sửa colors trong code
- Thay đổi Panel styles
- Cập nhật Table formats

## Liên hệ và đóng góp

### Báo lỗi
- Tạo issue trên GitHub
- Mô tả chi tiết lỗi
- Cung cấp log file

### Đóng góp
- Fork repository
- Tạo branch mới
- Submit pull request

### Phát triển
- Thêm tính năng mới
- Cải thiện hiệu suất
- Viết test cases

---

**Phiên bản:** 1.0.0  
**Ngày cập nhật:** 3 tháng 7, 2025  
**Tác giả:** AI Interview TuTran Studio
