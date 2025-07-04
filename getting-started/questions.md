# Hugging Face Questions Template

## Question 1: exercise_1
**Title:** Why build with Hugging Face?

**Description:** Hugging Face is a powerful platform that provides many benefits for building AI solutions, simplifying workflows and making advanced models accessible.

**Question:** Which of the following options is NOT a benefit of using Hugging Face for building AI solutions?

**Options:**
- Access to Pre-trained Models for Quick Prototyping
- Guaranteed Model Performance Across All Tasks
- End-to-End Support for ML Workflows
- Engaged Community and Extensive Documentation

**Correct Answer:** Guaranteed Model Performance Across All Tasks

**Explanation:** Hugging Face không đảm bảo hiệu suất model cho tất cả các tác vụ. Hiệu suất phụ thuộc vào nhiều yếu tố như dữ liệu, domain, và cách fine-tuning. Các lợi ích thực sự của Hugging Face bao gồm: truy cập model pre-trained, hỗ trợ end-to-end ML workflow, và cộng đồng mạnh mẽ.

---

## Question 2: exercise_2
**Title:** Hugging Face Hub Overview

**Description:** The Hugging Face Hub is a central platform that hosts machine learning models, datasets, and other resources for the AI community.

**Question:** What is the primary purpose of the Hugging Face Hub?

**Options:**
- Only hosting GPT models
- Providing a centralized platform for ML models and datasets
- Creating new AI algorithms
- Selling commercial AI software

**Correct Answer:** Providing a centralized platform for ML models and datasets

**Explanation:** Hugging Face Hub là nền tảng trung tâm để lưu trữ và chia sẻ các model ML, dataset, và tài nguyên khác. Nó không chỉ giới hạn ở GPT models mà hỗ trợ hàng triệu model khác nhau cho nhiều tác vụ khác nhau.

---

## Question 3: exercise_3
**Title:** Large Language Models (LLMs)

**Description:** Large Language Models are powerful AI models trained on massive datasets to understand and generate human-like text.

**Question:** Which of the following is NOT a popular LLM available on Hugging Face?

**Options:**
- GPT (OpenAI)
- Llama (Meta)
- BERT (Google)
- Excel (Microsoft)

**Correct Answer:** Excel (Microsoft)

**Explanation:** Excel là phần mềm bảng tính của Microsoft, không phải là Large Language Model. GPT, Llama, và BERT đều là các LLM nổi tiếng có sẵn trên Hugging Face Hub.

---

## Question 4: exercise_4
**Title:** Programmatic Access to Hugging Face

**Description:** Hugging Face provides APIs and libraries to programmatically access and interact with models and datasets.

**Question:** Which library is used to interact with Hugging Face Hub programmatically?

**Options:**
- tensorflow_hub
- huggingface_hub
- pytorch_hub
- sklearn_hub

**Correct Answer:** huggingface_hub

**Explanation:** Thư viện `huggingface_hub` là thư viện chính thức để tương tác với Hugging Face Hub qua API. Nó cung cấp các phương thức như `list_models()`, `list_datasets()` và nhiều chức năng khác.

---

## Question 5: exercise_5
**Title:** Navigating the Hugging Face Hub for Text Classification

**Description:** The Learning Analytics team at DataCamp needs to segment learner feedback into positive and negative categories. They are exploring the Hugging Face Hub to find a suitable text classification model and test it to ensure it meets their requirements.

**Question:** From the four statements below about exploring the Hugging Face Hub for text classification, which one is FALSE?

**Options:**
- A text classification model was selected to analyze learner feedback
- Sorting by Most Downloads highlighted trusted, widely used models
- The inference API returned a low positive response, suggesting that a different model might be needed
- Checking the Files and Versions tab ensured the model fit DataCamp's storage needs

**Correct Answer:** The inference API returned a low positive response, suggesting that a different model might be needed

**Explanation:** Đây là câu trả lời SAI vì inference API trả về "low positive response" không có nghĩa là cần thay đổi model. Một response có confidence thấp có thể do nhiều yếu tố như input text không rõ ràng, hoặc model đang phân loại đúng nhưng với độ tin cậy thấp. Các câu còn lại đều đúng: việc chọn text classification model, sắp xếp theo downloads để tìm model uy tín, và kiểm tra Files & Versions tab để đảm bảo kích thước phù hợp đều là các bước hợp lý trong quy trình tìm kiếm model.

---

## Question 6: exercise_6
**Title:** Model Selection Best Practices on Hugging Face Hub

**Description:** When selecting a model from Hugging Face Hub for production use, there are several important factors to consider to ensure the chosen model meets your project requirements.

**Question:** Which of the following is the MOST important factor when selecting a text classification model for production use?

**Options:**
- The model has the most stars and likes
- The model fits your specific use case and has good performance metrics
- The model was created by a well-known organization
- The model has the largest number of parameters

**Correct Answer:** The model fits your specific use case and has good performance metrics

**Explanation:** Yếu tố quan trọng nhất khi chọn model cho production là model phải phù hợp với use case cụ thể và có performance metrics tốt. Số lượng stars/likes, tên tuổi tổ chức, hay số lượng parameters đều có thể là dấu hiệu tốt nhưng không đảm bảo model sẽ hoạt động tốt cho tác vụ cụ thể của bạn. Luôn test model với dữ liệu thực tế và đánh giá performance trước khi deploy.

---

## Template for New Questions

```markdown
## Question X: exercise_X
**Title:** [Tiêu đề câu hỏi]

**Description:** [Mô tả ngữ cảnh của câu hỏi]

**Question:** [Câu hỏi chính]

**Options:**
- [Lựa chọn 1]
- [Lựa chọn 2]
- [Lựa chọn 3]
- [Lựa chọn 4]

**Correct Answer:** [Đáp án đúng]

**Explanation:** [Giải thích tại sao đáp án này đúng]

---
```

## Hướng dẫn thêm câu hỏi mới:

1. **Copy template** ở cuối file
2. **Thay thế các placeholder** [...] bằng nội dung thực tế
3. **Đảm bảo exercise_id** là duy nhất (exercise_7, exercise_8, ...)
4. **Chạy lại tool** để test câu hỏi mới

## Lưu ý:
- Mỗi câu hỏi phải có **4 lựa chọn**
- **Correct Answer** phải khớp chính xác với một trong các Options
- **Explanation** nên bằng tiếng Việt để dễ hiểu
- Sử dụng `---` để phân cách giữa các câu hỏi