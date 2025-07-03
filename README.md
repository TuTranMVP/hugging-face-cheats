# 🤖 AI Interview Agent - Hướng dẫn sử dụng

## Tổng quan
AI Interview Agent là một công cụ CLI chatbox tương tác để phân tích dữ liệu từ các file .md và mô phỏng phỏng vấn kiến thức về Hugging Face.

## Tính năng chính

### 🎯 Chế độ Interview (Phỏng vấn)
- Phân tích câu hỏi từ file .md
- Tạo phiên phỏng vấn tương tác
- Theo dõi điểm số và thống kê
- Đưa ra lời khuyên dựa trên kết quả

### 💬 Chế độ Chat (Trò chuyện)
- Trò chuyện tương tác với AI
- Trả lời câu hỏi dựa trên kiến thức đã tải
- Tìm kiếm thông tin liên quan
- Hỗ trợ chuyển đổi sang chế độ phỏng vấn

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

### 3. Cấu trúc thư viện
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

### Ví dụ sử dụng

#### 1. Chế độ phỏng vấn
```bash
python main.py getting-started/questions.md --mode interview
```

#### 2. Chế độ chat
```bash
python main.py getting-started/questions.md getting-started/introduction.md --mode chat
```

#### 3. Cả hai chế độ
```bash
python main.py *.md --mode both
```

#### 4. Giới hạn số câu hỏi
```bash
python main.py questions.md --mode interview --limit 5
```

#### 5. Xáo trộn câu hỏi
```bash
python main.py questions.md --mode interview --shuffle
```

#### 6. Chế độ verbose
```bash
python main.py *.md --mode both --verbose
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
**Tác giả:** AI Interview Agent Team
