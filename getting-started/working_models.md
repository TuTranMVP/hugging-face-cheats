# Làm việc với Models từ Hugging Face Hub

## 📚 Tổng quan

Hugging Face Hub là nền tảng lưu trữ hàng nghìn mô hình AI pre-trained (đã được huấn luyện sẵn) cho các tác vụ xử lý văn bản, hình ảnh và âm thanh. Thư viện **Transformers** giúp đơn giản hóa việc sử dụng các mô hình này.

## 🔍 Khám phá Hugging Face Hub

### Truy cập Hub
- **Website chính**: [huggingface.co](https://huggingface.co/)
- **Danh sách models**: [huggingface.co/models](https://huggingface.co/models)

### Tính năng tìm kiếm
- **Lọc theo task**: Text classification, object detection, summarization...
- **Lọc theo ngôn ngữ**: Tiếng Việt, Tiếng Anh, v.v.
- **Lọc theo thư viện**: PyTorch, TensorFlow, JAX...
- **Trạng thái**: Open-source (miễn phí) hoặc có phí

### Model Cards
Mỗi model có thông tin chi tiết:
- ✅ Mô tả và cách sử dụng
- ✅ Tác giả phát triển
- ✅ Giấy phép (license)
- ✅ Ngôn ngữ hỗ trợ
- ✅ Kích thước file

## 🚀 Sử dụng Pipelines

### Pipeline cơ bản
```python
from transformers import pipeline

# Tạo pipeline cho phân loại văn bản
classifier = pipeline("text-classification", 
                     model="distilbert-base-uncased-finetuned-sst-2-english")

# Sử dụng
result = classifier("DataCamp is awesome!")
# Kết quả: [{'label': 'POSITIVE', 'score': 0.99}]
```

### Pipeline nâng cao
```python
# Text generation với tham số tùy chỉnh
generator = pipeline("text-generation", model="gpt2")

result = generator("Artificial intelligence is", 
                  max_length=10,           # Giới hạn độ dài
                  num_return_sequences=2)  # Tạo 2 kết quả khác nhau
```

## 📁 Lưu trữ Models

### Khi nào cần lưu model?
- 🔒 **Sử dụng offline**: Không có internet
- 🛠️ **Fine-tuning**: Tùy chỉnh model cho dự án cụ thể
- 🏢 **Production**: Triển khai quy mô lớn
- 💾 **Kiểm soát storage**: Quản lý dung lượng

### Cách lưu model
```python
from transformers import pipeline

# Tạo pipeline
classifier = pipeline("text-classification", 
                     model="distilbert-base-uncased-finetuned-sst-2-english")

# Lưu model và tokenizer
classifier.save_pretrained("./my_model")

# Load lại từ file local
classifier_local = pipeline("text-classification", model="./my_model")
```

## ⚠️ Lưu ý quan trọng

### Kích thước Model
- **DistilBERT**: ~1GB
- **BERT Large**: ~3GB  
- **GPT-3 style models**: 10GB+
- ➡️ Kiểm tra tab "Files and Versions" trước khi tải

### Tokens
- **Token**: Đơn vị nhỏ nhất mà model xử lý (từ, ký tự hoặc subword)
- **max_length**: Giới hạn số token đầu ra
- **Ví dụ**: "Hello world" có thể được chia thành 2-3 tokens

### Performance Tips
- 🚀 Models được cache tự động khi sử dụng pipeline
- 💡 Chỉ save local khi thực sự cần thiết
- 🔄 Kết quả text generation sẽ khác nhau mỗi lần chạy

## 📋 Checklist cho Developer

- [ ] Tìm hiểu model phù hợp trên Hub
- [ ] Đọc model card để hiểu cách sử dụng
- [ ] Kiểm tra license và requirements
- [ ] Test với pipeline trước khi integrate
- [ ] Cân nhắc việc save local dựa trên use case
- [ ] Monitor performance và kích thước model

## 🔗 Tài liệu tham khảo

- [Hugging Face Transformers Documentation](https://github.com/huggingface/transformers)
- [Model Hub](https://huggingface.co/models)
- [Pipeline Documentation](https://huggingface.co/docs/transformers/main_classes/pipelines)
