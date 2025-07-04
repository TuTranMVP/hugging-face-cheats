# Câu Hỏi Text Classification Pipeline

## Câu Hỏi 1: exercise_1
**Tiêu đề:** Kiểm Tra Ngữ Pháp với Text Classification

**Mô tả:** Text classification có thể được sử dụng cho nhiều tác vụ khác nhau bao gồm phát hiện lỗi ngữ pháp. Điều này bao gồm việc sử dụng các mô hình pre-trained để xác định xem một câu có chứa lỗi ngữ pháp hay không.

**Câu hỏi:** Khi sử dụng text-classification pipeline để kiểm tra ngữ pháp, cách nào sau đây là đúng để tạo một grammar checker pipeline?

**Các lựa chọn:**
- `pipeline(task="grammar-check", model="abdulmatinomotoso/English_Grammar_Checker")`
- `pipeline(task="text-classification", model="abdulmatinomotoso/English_Grammar_Checker")`
- `pipeline(task="text-generation", model="abdulmatinomotoso/English_Grammar_Checker")`
- `pipeline(task="sentiment-analysis", model="abdulmatinomotoso/English_Grammar_Checker")`

**Đáp án đúng:** `pipeline(task="text-classification", model="abdulmatinomotoso/English_Grammar_Checker")`

**Giải thích:** Kiểm tra ngữ pháp là một ứng dụng cụ thể của text classification, trong đó mô hình phân loại văn bản thành đúng ngữ pháp hoặc sai ngữ pháp. Tham số task phải là "text-classification" với mô hình kiểm tra ngữ pháp phù hợp.

---

## Câu Hỏi 2: exercise_2
**Tiêu đề:** Question Natural Language Inference (QNLI)

**Mô tả:** QNLI là một tác vụ text classification xác định xem một tiền đề cho trước có chứa đủ thông tin để trả lời một câu hỏi được đặt ra hay không.

**Câu hỏi:** QNLI (Question Natural Language Inference) đánh giá điều gì cụ thể?

**Các lựa chọn:**
- Liệu một câu hỏi có đúng ngữ pháp hay không
- Liệu một tiền đề có chứa đủ thông tin để trả lời câu hỏi hay không
- Liệu hai câu có cùng nghĩa hay không
- Liệu một văn bản có tính tích cực hay tiêu cực trong cảm xúc

**Đáp án đúng:** Liệu một tiền đề có chứa đủ thông tin để trả lời câu hỏi hay không

**Giải thích:** QNLI đánh giá xem một tiền đề/văn bản cho trước có chứa đủ thông tin để trả lời một câu hỏi được đặt ra hay không. Nó xác định xem câu trả lời có thể được tìm thấy trong văn bản được cung cấp, làm cho nó hữu ích cho các tác vụ đọc hiểu và truy xuất thông tin.

---

## Câu Hỏi 3: exercise_3
**Tiêu đề:** Zero-Shot Classification

**Mô tả:** Zero-shot classification cho phép các mô hình phân loại văn bản thành các danh mục được định nghĩa trước mà không cần huấn luyện trước trên những danh mục cụ thể đó.

**Câu hỏi:** Trong zero-shot classification, điều gì cần thiết để phân loại văn bản vào các danh mục mà mô hình chưa được huấn luyện cụ thể?

**Các lựa chọn:**
- Fine-tuning mô hình trên các danh mục đích
- Cung cấp văn bản và danh sách các nhãn danh mục được định nghĩa trước
- Huấn luyện một mô hình mới từ đầu
- Chỉ sử dụng dữ liệu huấn luyện gốc

**Đáp án đúng:** Cung cấp văn bản và danh sách các nhãn danh mục được định nghĩa trước

**Giải thích:** Zero-shot classification hoạt động bằng cách cung cấp văn bản đầu vào cùng với danh sách các nhãn/danh mục ứng viên. Mô hình sau đó có thể phân loại văn bản vào một trong những danh mục này mà không cần huấn luyện cụ thể trên những danh mục đó, sử dụng hiểu biết ngôn ngữ tổng quát của nó.

---

## Câu Hỏi 4: exercise_4
**Tiêu đề:** Các Tác Vụ Text Classification Pipeline

**Mô tả:** Text-classification pipeline trong Hugging Face có thể xử lý nhiều tác vụ khác nhau bằng cách sử dụng các mô hình pre-trained được tối ưu hóa cho các mục tiêu phân loại cụ thể.

**Câu hỏi:** Điều nào sau đây thường KHÔNG được coi là một tác vụ text classification?

**Các lựa chọn:**
- Phân tích cảm xúc (tích cực/tiêu cực)
- Phát hiện spam (spam/không spam)
- Phát hiện lỗi ngữ pháp
- Tóm tắt văn bản

**Đáp án đúng:** Tóm tắt văn bản

**Giải thích:** Tóm tắt văn bản là một tác vụ sinh văn bản (text generation), không phải tác vụ phân loại. Nó bao gồm việc tạo ra một phiên bản ngắn hơn của văn bản đầu vào thay vì gán nó vào các danh mục được định nghĩa trước. Các tác vụ text classification bao gồm việc phân loại văn bản đầu vào thành các nhãn hoặc lớp rời rạc.

---

## Câu Hỏi 5: exercise_5
**Tiêu đề:** Lựa Chọn Mô Hình cho Tác Vụ Cụ Thể

**Mô tả:** Các mô hình text classification khác nhau được tối ưu hóa cho các tác vụ khác nhau. Việc chọn đúng mô hình là quan trọng để đạt được hiệu suất tốt trên các mục tiêu phân loại cụ thể.

**Câu hỏi:** Khi thực hiện QNLI (Question Natural Language Inference), loại kiến trúc mô hình nào được sử dụng phổ biến nhất?

**Các lựa chọn:**
- Các mô hình sinh như GPT
- Các mô hình cross-encoder như "cross-encoder/qnli-electra-base"
- Các mô hình phân loại hình ảnh
- Các mô hình nhận dạng giọng nói

**Đáp án đúng:** Các mô hình cross-encoder như "cross-encoder/qnli-electra-base"

**Giải thích:** Các mô hình cross-encoder được thiết kế đặc biệt cho các tác vụ đòi hỏi hiểu biết mối quan hệ giữa hai đoạn văn bản (câu hỏi và tiền đề). Chúng xử lý cả hai đầu vào đồng thời, làm cho chúng lý tưởng cho các tác vụ QNLI nơi mô hình cần xác định xem một tiền đề có trả lời được câu hỏi hay không.

---

## Câu Hỏi 6: exercise_6
**Tiêu đề:** Định Dạng Đầu Ra Pipeline

**Mô tả:** Các text classification pipeline trả về đầu ra có cấu trúc chứa các nhãn và điểm số tin cậy, giúp hiểu được các dự đoán của mô hình.

**Câu hỏi:** Trong zero-shot classification, thông tin nào thường được bao gồm trong đầu ra của pipeline?

**Các lựa chọn:**
- Chỉ nhãn được dự đoán
- Chỉ điểm số tin cậy
- Cả nhãn được xếp hạng theo độ tin cậy và điểm số tương ứng của chúng
- Chỉ văn bản đầu vào gốc

**Đáp án đúng:** Cả nhãn được xếp hạng theo độ tin cậy và điểm số tương ứng của chúng

**Giải thích:** Đầu ra zero-shot classification thường bao gồm danh sách các nhãn được xếp hạng theo điểm số tin cậy và điểm số xác suất tương ứng của chúng. Điều này cho phép người dùng không chỉ thấy dự đoán hàng đầu mà còn các phân loại thay thế và mức độ tin cậy của chúng.

---

## Câu Hỏi 7: exercise_7
**Tiêu đề:** Gán Danh Mục Động

**Mô tả:** Gán danh mục động trong zero-shot classification cho phép phân loại văn bản linh hoạt mà không cần huấn luyện lại mô hình cho các danh mục mới.

**Câu hỏi:** Ưu điểm chính của gán danh mục động trong zero-shot classification là gì?

**Các lựa chọn:**
- Nó đòi hỏi ít tài nguyên tính toán hơn
- Nó có thể phân loại văn bản vào các danh mục mới mà không cần huấn luyện lại mô hình
- Nó luôn cung cấp kết quả chính xác 100%
- Nó chỉ hoạt động với các danh mục được định nghĩa trước

**Đáp án đúng:** Nó có thể phân loại văn bản vào các danh mục mới mà không cần huấn luyện lại mô hình

**Giải thích:** Ưu điểm chính của gán danh mục động là tính linh hoạt - bạn có thể định nghĩa các danh mục mới một cách nhanh chóng mà không cần huấn luyện lại hoặc fine-tune mô hình. Điều này làm cho nó cực kỳ hữu ích cho các ứng dụng nơi các danh mục có thể thay đổi thường xuyên hoặc khi bạn cần thử nghiệm với các sơ đồ phân loại khác nhau.

---

## Câu Hỏi 8: exercise_8
**Tiêu đề:** Ứng Dụng Text Classification

**Mô tả:** Text classification có nhiều ứng dụng thực tế trên các lĩnh vực khác nhau, từ kiểm duyệt nội dung đến tổ chức thông tin.

**Câu hỏi:** Tình huống nào sau đây sẽ được hưởng lợi nhiều nhất từ việc sử dụng text classification pipeline?

**Các lựa chọn:**
- Tạo ra các câu chuyện sáng tạo
- Tự động phân loại vé hỗ trợ khách hàng
- Dịch văn bản giữa các ngôn ngữ
- Chuyển đổi giọng nói thành văn bản

**Đáp án đúng:** Tự động phân loại vé hỗ trợ khách hàng

**Giải thích:** Tự động phân loại vé hỗ trợ khách hàng là một trường hợp sử dụng text classification hoàn hảo. Nó bao gồm việc lấy văn bản đầu vào (vé hỗ trợ) và gán nó vào các danh mục được định nghĩa trước (như "thanh toán", "kỹ thuật", "yêu cầu chung"), giúp định tuyến vé đến các phòng ban phù hợp và ưu tiên phản hồi.

---

## Tóm Tắt Các Khái Niệm Chính

### Từ Khóa Chính:
- **Text Classification**: Quá trình gán nhãn văn bản đầu vào vào các danh mục được định nghĩa trước
- **Pipeline**: Giao diện cấp cao của Hugging Face cho các tác vụ NLP
- **QNLI (Question Natural Language Inference)**: Tác vụ xác định xem tiền đề có trả lời được câu hỏi
- **Zero-shot Classification**: Phân loại văn bản mà không cần huấn luyện cụ thể cho tác vụ
- **Grammar Checking**: Sử dụng phân loại để phát hiện lỗi ngữ pháp
- **Cross-encoder**: Kiến trúc mô hình cho các tác vụ cặp văn bản
- **Dynamic Category Assignment**: Phân loại linh hoạt mà không cần huấn luyện lại

### Khái Niệm Kỹ Thuật:
- Lựa chọn mô hình cho các tác vụ cụ thể
- Cấu hình pipeline và tham số
- Diễn giải đầu ra (nhãn và điểm số)
- Sử dụng mô hình pre-trained
- Tối ưu hóa mô hình cụ thể cho tác vụ

### Ứng Dụng Thực Tế:
- Phân tích cảm xúc
- Phát hiện spam
- Phát hiện lỗi ngữ pháp
- Xác thực trả lời câu hỏi
- Phân loại nội dung
- Định tuyến vé hỗ trợ khách hàng
