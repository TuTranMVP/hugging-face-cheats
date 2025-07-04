# Mẫu Câu Hỏi Hugging Face

## Câu Hỏi 1: exercise_1
**Tiêu đề:** Tại sao nên xây dựng với Hugging Face?

**Mô tả:** Hugging Face là một nền tảng mạnh mẽ cung cấp nhiều lợi ích cho việc xây dựng các giải pháp AI, đơn giản hóa quy trình làm việc và làm cho các mô hình tiên tiến trở nên dễ tiếp cận.

**Câu hỏi:** Điều nào sau đây KHÔNG phải là lợi ích của việc sử dụng Hugging Face để xây dựng giải pháp AI?

**Các lựa chọn:**
- Truy cập các Mô hình Pre-trained để Tạo Prototype Nhanh
- Đảm bảo Hiệu suất Mô hình cho Tất cả Các Tác vụ
- Hỗ trợ End-to-End cho Quy trình ML
- Cộng đồng Tham gia Tích cực và Tài liệu Phong phú

**Đáp án đúng:** Đảm bảo Hiệu suất Mô hình cho Tất cả Các Tác vụ

**Giải thích:** Hugging Face không đảm bảo hiệu suất model cho tất cả các tác vụ. Hiệu suất phụ thuộc vào nhiều yếu tố như dữ liệu, domain, và cách fine-tuning. Các lợi ích thực sự của Hugging Face bao gồm: truy cập model pre-trained, hỗ trợ end-to-end ML workflow, và cộng đồng mạnh mẽ.

---

## Câu Hỏi 2: exercise_2
**Tiêu đề:** Tổng quan về Hugging Face Hub

**Mô tả:** Hugging Face Hub là một nền tảng trung tâm lưu trữ các mô hình machine learning, dataset, và các tài nguyên khác cho cộng đồng AI.

**Câu hỏi:** Mục đích chính của Hugging Face Hub là gì?

**Các lựa chọn:**
- Chỉ lưu trữ các mô hình GPT
- Cung cấp nền tảng tập trung cho các mô hình ML và dataset
- Tạo ra các thuật toán AI mới
- Bán phần mềm AI thương mại

**Đáp án đúng:** Cung cấp nền tảng tập trung cho các mô hình ML và dataset

**Giải thích:** Hugging Face Hub là nền tảng trung tâm để lưu trữ và chia sẻ các model ML, dataset, và tài nguyên khác. Nó không chỉ giới hạn ở GPT models mà hỗ trợ hàng triệu model khác nhau cho nhiều tác vụ khác nhau.

---

## Câu Hỏi 3: exercise_3
**Tiêu đề:** Mô hình Ngôn ngữ Lớn (LLMs)

**Mô tả:** Các Mô hình Ngôn ngữ Lớn là những mô hình AI mạnh mẽ được huấn luyện trên các bộ dữ liệu khổng lồ để hiểu và tạo ra văn bản giống con người.

**Câu hỏi:** Điều nào sau đây KHÔNG phải là một LLM phổ biến có sẵn trên Hugging Face?

**Các lựa chọn:**
- GPT (OpenAI)
- Llama (Meta)
- BERT (Google)
- Excel (Microsoft)

**Đáp án đúng:** Excel (Microsoft)

**Giải thích:** Excel là phần mềm bảng tính của Microsoft, không phải là Large Language Model. GPT, Llama, và BERT đều là các LLM nổi tiếng có sẵn trên Hugging Face Hub.

---

## Câu Hỏi 4: exercise_4
**Tiêu đề:** Truy cập Lập trình đến Hugging Face

**Mô tả:** Hugging Face cung cấp các API và thư viện để truy cập và tương tác với các mô hình và dataset một cách lập trình.

**Câu hỏi:** Thư viện nào được sử dụng để tương tác với Hugging Face Hub một cách lập trình?

**Các lựa chọn:**
- tensorflow_hub
- huggingface_hub
- pytorch_hub
- sklearn_hub

**Đáp án đúng:** huggingface_hub

**Giải thích:** Thư viện `huggingface_hub` là thư viện chính thức để tương tác với Hugging Face Hub qua API. Nó cung cấp các phương thức như `list_models()`, `list_datasets()` và nhiều chức năng khác.

---

## Câu Hỏi 5: exercise_5
**Tiêu đề:** Điều hướng Hugging Face Hub cho Phân loại Văn bản

**Mô tả:** Nhóm Learning Analytics tại DataCamp cần phân loại phản hồi của học viên thành các danh mục tích cực và tiêu cực. Họ đang khám phá Hugging Face Hub để tìm một mô hình phân loại văn bản phù hợp và kiểm tra nó để đảm bảo đáp ứng yêu cầu của họ.

**Câu hỏi:** Trong bốn tuyên bố dưới đây về việc khám phá Hugging Face Hub cho phân loại văn bản, điều nào là SAI?

**Các lựa chọn:**
- Một mô hình phân loại văn bản đã được chọn để phân tích phản hồi của học viên
- Sắp xếp theo Lượt Tải xuống Nhiều nhất làm nổi bật các mô hình đáng tin cậy, được sử dụng rộng rãi
- API suy luận trả về phản hồi tích cực thấp, gợi ý rằng có thể cần một mô hình khác
- Kiểm tra tab Files và Versions đảm bảo mô hình phù hợp với nhu cầu lưu trữ của DataCamp

**Đáp án đúng:** API suy luận trả về phản hồi tích cực thấp, gợi ý rằng có thể cần một mô hình khác

**Giải thích:** Đây là câu trả lời SAI vì inference API trả về "low positive response" không có nghĩa là cần thay đổi model. Một response có confidence thấp có thể do nhiều yếu tố như input text không rõ ràng, hoặc model đang phân loại đúng nhưng với độ tin cậy thấp. Các câu còn lại đều đúng: việc chọn text classification model, sắp xếp theo downloads để tìm model uy tín, và kiểm tra Files & Versions tab để đảm bảo kích thước phù hợp đều là các bước hợp lý trong quy trình tìm kiếm model.

---

## Câu Hỏi 6: exercise_6
**Tiêu đề:** Thực hành Tốt nhất cho Lựa chọn Mô hình trên Hugging Face Hub

**Mô tả:** Khi lựa chọn một mô hình từ Hugging Face Hub để sử dụng trong production, có nhiều yếu tố quan trọng cần xem xét để đảm bảo mô hình được chọn đáp ứng yêu cầu dự án của bạn.

**Câu hỏi:** Điều nào sau đây là yếu tố QUAN TRỌNG NHẤT khi lựa chọn mô hình phân loại văn bản để sử dụng trong production?

**Các lựa chọn:**
- Mô hình có nhiều sao và lượt thích nhất
- Mô hình phù hợp với trường hợp sử dụng cụ thể của bạn và có chỉ số hiệu suất tốt
- Mô hình được tạo bởi một tổ chức nổi tiếng
- Mô hình có số lượng tham số lớn nhất

**Đáp án đúng:** Mô hình phù hợp với trường hợp sử dụng cụ thể của bạn và có chỉ số hiệu suất tốt

**Giải thích:** Yếu tố quan trọng nhất khi chọn model cho production là model phải phù hợp với use case cụ thể và có performance metrics tốt. Số lượng stars/likes, tên tuổi tổ chức, hay số lượng parameters đều có thể là dấu hiệu tốt nhưng không đảm bảo model sẽ hoạt động tốt cho tác vụ cụ thể của bạn. Luôn test model với dữ liệu thực tế và đánh giá performance trước khi deploy.

---

## Mẫu cho Câu hỏi Mới

```markdown
## Câu Hỏi X: exercise_X
**Tiêu đề:** [Tiêu đề câu hỏi]

**Mô tả:** [Mô tả ngữ cảnh của câu hỏi]

**Câu hỏi:** [Câu hỏi chính]

**Các lựa chọn:**
- [Lựa chọn 1]
- [Lựa chọn 2]
- [Lựa chọn 3]
- [Lựa chọn 4]

**Đáp án đúng:** [Đáp án đúng]

**Giải thích:** [Giải thích tại sao đáp án này đúng]

---
```

## Hướng dẫn thêm câu hỏi mới:

1. **Copy template** ở cuối file
2. **Thay thế các placeholder** [...] bằng nội dung thực tế
3. **Đảm bảo exercise_id** là duy nhất (exercise_7, exercise_8, ...)
4. **Chạy lại tool** để test câu hỏi mới

## Lưu ý:
- Mỗi câu hỏi phải có **4 lựa chọn**
- **Đáp án đúng** phải khớp chính xác với một trong các Lựa chọn
- **Giải thích** nên bằng tiếng Việt để dễ hiểu
- Sử dụng `---` để phân cách giữa các câu hỏi
