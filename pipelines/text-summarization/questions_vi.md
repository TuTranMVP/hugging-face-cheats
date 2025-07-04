# Câu Hỏi Text Summarization Pipeline

## Câu Hỏi 1: exercise_1
**Tiêu đề:** Cơ Bản về Text Summarization

**Mô tả:** Text summarization rút gọn văn bản lớn thành nội dung dễ quản lý, giúp người đọc nhanh chóng nắm bắt các điểm chính từ các bài viết hoặc tài liệu dài.

**Câu hỏi:** Mục đích chính của text summarization là gì?

**Các lựa chọn:**
- Dịch văn bản giữa các ngôn ngữ khác nhau
- Rút gọn văn bản lớn thành nội dung dễ quản lý trong khi bảo tồn thông tin chính
- Tạo ra nội dung hoàn toàn mới không liên quan đến bản gốc
- Kiểm tra lỗi ngữ pháp và chính tả

**Đáp án đúng:** Rút gọn văn bản lớn thành nội dung dễ quản lý trong khi bảo tồn thông tin chính

**Giải thích:** Text summarization được thiết kế đặc biệt để nén các tài liệu hoặc bài viết dài thành các phiên bản ngắn hơn giữ lại thông tin quan trọng nhất, giúp người đọc dễ dàng hiểu các điểm chính mà không cần đọc toàn bộ văn bản gốc.

---

## Câu Hỏi 2: exercise_2
**Tiêu đề:** Phân Loại Abstractive vs Extractive Summarization

**Mô tả:** Có hai loại summarization chính: extractive (chọn các câu chính từ văn bản gốc) và abstractive (tạo ra các câu mới tóm tắt ý chính).

**Câu hỏi:** Sự khác biệt chính giữa extractive và abstractive summarization là gì?

**Các lựa chọn:**
- Extractive tạo câu mới, abstractive chọn câu có sẵn
- Extractive chọn các câu chính từ văn bản gốc, abstractive tạo ra các câu mới
- Cả hai phương pháp hoạt động giống hệt nhau
- Extractive chỉ dành cho văn bản ngắn, abstractive cho văn bản dài

**Đáp án đúng:** Extractive chọn các câu chính từ văn bản gốc, abstractive tạo ra các câu mới

**Giải thích:** Extractive summarization xác định và chọn những câu quan trọng nhất trực tiếp từ văn bản gốc, trong khi abstractive summarization sử dụng sinh ngôn ngữ tự nhiên để tạo ra các câu mới nắm bắt được ý chính theo cách diễn đạt lại.

---

## Câu Hỏi 3: exercise_3
**Tiêu đề:** Tạo Pipeline cho Summarization

**Mô tả:** Việc tạo một summarization pipeline bao gồm việc chỉ định task và model một cách phù hợp cho phương pháp summarization mong muốn.

**Câu hỏi:** Cách đúng để tạo một abstractive summarization pipeline sử dụng model cnicu/t5-small-booksum là gì?

**Các lựa chọn:**
- `pipeline(task="text-generation", model="cnicu/t5-small-booksum")`
- `pipeline(task="summarization", model="cnicu/t5-small-booksum")`
- `pipeline(task="text-classification", model="cnicu/t5-small-booksum")`
- `pipeline(task="translation", model="cnicu/t5-small-booksum")`

**Đáp án đúng:** `pipeline(task="summarization", model="cnicu/t5-small-booksum")`

**Giải thích:** Đối với các tác vụ summarization, tham số task phải là "summarization" bất kể bạn đang thực hiện extractive hay abstractive summarization. Tham số model chỉ định model pre-trained nào sử dụng cho tác vụ.

---

## Câu Hỏi 4: exercise_4
**Tiêu đề:** Hiểu Cấu Trúc Đầu Ra Pipeline

**Mô tả:** Các summarization pipeline trả về đầu ra có cấu trúc chứa văn bản tóm tắt trong một định dạng cụ thể.

**Câu hỏi:** Làm thế nào để truy cập văn bản tóm tắt từ đầu ra của summarization pipeline?

**Các lựa chọn:**
- `summary_text[0]`
- `summary_text['summary_text']`
- `summary_text[0]['summary_text']`
- `summary_text.summary_text`

**Đáp án đúng:** `summary_text[0]['summary_text']`

**Giải thích:** Các summarization pipeline trả về một danh sách các dictionary. Để truy cập văn bản tóm tắt thực tế, bạn cần index phần tử đầu tiên [0] và sau đó truy cập key 'summary_text' từ dictionary đó.

---

## Câu Hỏi 5: exercise_5
**Tiêu đề:** Tham Số Kiểm Soát Độ Dài

**Mô tả:** Các tham số min_length và max_length rất quan trọng để kiểm soát độ dài của các bản tóm tắt được tạo ra để đáp ứng các yêu cầu cụ thể.

**Câu hỏi:** Các tham số min_length và max_length kiểm soát điều gì trong summarization pipeline?

**Các lựa chọn:**
- Độ dài của văn bản đầu vào tính bằng ký tự
- Độ dài của đầu ra tóm tắt tính bằng token
- Thời gian xử lý của model
- Điểm chất lượng của bản tóm tắt

**Đáp án đúng:** Độ dài của đầu ra tóm tắt tính bằng token

**Giải thích:** Các tham số min_length và max_length kiểm soát cụ thể độ dài của bản tóm tắt được tạo ra đo bằng token. Điều này cho phép bạn đảm bảo các bản tóm tắt nằm trong các ràng buộc độ dài mong muốn cho các trường hợp sử dụng khác nhau.

---

## Câu Hỏi 6: exercise_6
**Tiêu đề:** Cấu Hình Tóm Tắt Ngắn

**Mô tả:** Tạo các bản tóm tắt rất ngắn đòi hỏi phải đặt các ràng buộc độ dài phù hợp để đảm bảo đầu ra súc tích.

**Câu hỏi:** Để tạo một bản tóm tắt rất ngắn (1-10 token), bạn nên sử dụng cấu hình tham số nào?

**Các lựa chọn:**
- `min_length=1, max_length=10`
- `min_length=10, max_length=1`
- `min_length=0, max_length=100`
- `min_length=50, max_length=150`

**Đáp án đúng:** `min_length=1, max_length=10`

**Giải thích:** Đối với bản tóm tắt ngắn, bạn đặt min_length ở giá trị thấp (1) và max_length ở số nhỏ (10). Điều này đảm bảo bản tóm tắt sẽ có độ dài từ 1 đến 10 token, tạo ra đầu ra rất súc tích.

---

## Câu Hỏi 7: exercise_7
**Tiêu đề:** Cấu Hình Tóm Tắt Dài

**Mô tả:** Đối với các bản tóm tắt chi tiết hơn, bạn cần đặt các khoảng độ dài rộng hơn để nắm bắt thông tin toàn diện hơn.

**Câu hỏi:** Cài đặt tham số nào sẽ phù hợp để tạo ra một bản tóm tắt chi tiết?

**Các lựa chọn:**
- `min_length=1, max_length=10`
- `min_length=50, max_length=150`
- `min_length=150, max_length=50`
- `min_length=0, max_length=5`

**Đáp án đúng:** `min_length=50, max_length=150`

**Giải thích:** Đối với bản tóm tắt chi tiết, bạn cần một khoảng rộng hơn với các giá trị cao hơn. Đặt min_length=50 và max_length=150 đảm bảo bản tóm tắt chứa đủ chi tiết (ít nhất 50 token) trong khi không trở nên quá dài dòng (tối đa 150 token).

---

## Câu Hỏi 8: exercise_8
**Tiêu đề:** Ứng Dụng Thực Tế và Use Cases

**Mô tả:** Hiểu khi nào sử dụng các độ dài tóm tắt khác nhau là quan trọng cho các ứng dụng thực tế trong nhiều lĩnh vực khác nhau.

**Câu hỏi:** Trong tình huống nào bạn có khả năng sử dụng bản tóm tắt rất ngắn (1-10 token) nhất?

**Các lựa chọn:**
- Phân tích bài báo nghiên cứu học thuật
- Xem xét tài liệu pháp lý
- Tiêu đề bài đăng mạng xã hội hoặc thẻ tag
- Tóm tắt báo cáo kinh doanh chi tiết

**Đáp án đúng:** Tiêu đề bài đăng mạng xã hội hoặc thẻ tag

**Giải thích:** Bản tóm tắt rất ngắn (1-10 token) lý tưởng để tạo tiêu đề súc tích, thẻ tag, hoặc mô tả ngắn gọn cho mạng xã hội nơi không gian bị hạn chế và người dùng cần nhanh chóng nắm bắt được bản chất của nội dung.

---

## Câu Hỏi 9: exercise_9
**Tiêu đề:** Lựa Chọn Model cho Summarization

**Mô tả:** Các model khác nhau được tối ưu hóa cho các loại tác vụ summarization và lĩnh vực nội dung khác nhau.

**Câu hỏi:** Điều gì làm cho model "cnicu/t5-small-booksum" đặc biệt phù hợp cho text summarization?

**Các lựa chọn:**
- Nó được thiết kế đặc biệt cho các tác vụ dịch thuật
- Nó được huấn luyện trên dữ liệu tóm tắt sách, làm cho nó tốt cho việc tóm tắt văn bản dài
- Nó chỉ hoạt động với văn bản ngắn
- Nó là model nhanh nhất có sẵn

**Đáp án đúng:** Nó được huấn luyện trên dữ liệu tóm tắt sách, làm cho nó tốt cho việc tóm tắt văn bản dài

**Giải thích:** Model "cnicu/t5-small-booksum" được huấn luyện cụ thể trên dữ liệu tóm tắt sách, điều này làm cho nó đặc biệt hiệu quả trong việc xử lý văn bản dài và tạo ra các bản tóm tắt mạch lạc nắm bắt được các chủ đề chính và ý tưởng từ nội dung mở rộng.

---

## Câu Hỏi 10: exercise_10
**Tiêu đề:** So Sánh Độ Dài Gốc và Tóm Tắt

**Mô tả:** Đo lường hiệu quả của summarization thường bao gồm việc so sánh độ dài của văn bản gốc và văn bản tóm tắt.

**Câu hỏi:** Lợi ích chính của việc so sánh độ dài văn bản gốc với độ dài tóm tắt là gì?

**Các lựa chọn:**
- Để kiểm tra lỗi ngữ pháp
- Để đo tỷ lệ nén và hiệu quả summarization
- Để xác định độ chính xác dịch thuật
- Để xác thực việc huấn luyện model

**Đáp án đúng:** Để đo tỷ lệ nén và hiệu quả summarization

**Giải thích:** So sánh độ dài giúp bạn hiểu tỷ lệ nén đạt được bởi quá trình summarization. Chỉ số này rất quan trọng để đánh giá xem model có hiệu quả nén nội dung và đáp ứng các yêu cầu độ dài cụ thể của bạn cho các ứng dụng khác nhau hay không.

---

## Tóm Tắt Các Khái Niệm Chính

### Từ Khóa Chính:
- **Text Summarization**: Quá trình rút gọn văn bản lớn trong khi bảo tồn thông tin chính
- **Abstractive Summarization**: Tạo ra các câu mới tóm tắt ý chính
- **Extractive Summarization**: Chọn các câu chính từ văn bản gốc
- **Pipeline**: Giao diện Hugging Face cho các tác vụ summarization
- **min_length/max_length**: Tham số kiểm soát độ dài tóm tắt tính bằng token
- **Token**: Đơn vị cơ bản của xử lý văn bản trong các model ngôn ngữ
- **Model Selection**: Chọn model phù hợp cho nhu cầu summarization cụ thể

### Khái Niệm Kỹ Thuật:
- Cấu hình pipeline và tham số
- Cấu trúc đầu ra và truy cập dữ liệu
- Cơ chế kiểm soát độ dài
- Khả năng cụ thể của model
- Đo lường hiệu suất thông qua so sánh độ dài

### Ứng Dụng Thực Tế:
- Xử lý tài liệu dài
- Nén nội dung cho ràng buộc lưu trữ
- Trích xuất thông tin nhanh
- Tạo nội dung mạng xã hội
- Phân tích tài liệu học thuật và kinh doanh
