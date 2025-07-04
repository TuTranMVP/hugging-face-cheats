# Hugging Face Datasets

## 📚 Tổng quan

Hugging Face Hub cung cấp hàng nghìn datasets được cộng đồng tuyển chọn cho nhiều tác vụ và lĩnh vực khác nhau. Thư viện **datasets** giúp truy cập, tải xuống, xử lý và chia sẻ dữ liệu chỉ với vài dòng code.

## 🔍 Khám phá Datasets

### Truy cập Hub
- **Datasets Hub**: [huggingface.co/datasets](https://huggingface.co/datasets)
- **Documentation**: [huggingface.co/docs/datasets](https://huggingface.co/docs/datasets/index)

### Tính năng tìm kiếm
- **Lọc theo task**: Text classification, sentiment analysis, translation...
- **Lọc theo ngôn ngữ**: Tiếng Việt, Tiếng Anh, đa ngôn ngữ...
- **Lọc theo kích thước**: Nhỏ (<1MB), trung bình (1MB-1GB), lớn (>1GB)
- **Lọc theo license**: MIT, Apache, Creative Commons...

### Dataset Cards
Mỗi dataset có thông tin chi tiết:
- ✅ **Dataset path**: Định danh duy nhất (vd: `imdb`, `squad`)
- ✅ **Description**: Mô tả mục đích và nội dung
- ✅ **Dataset structure**: Cấu trúc cột và kiểu dữ liệu
- ✅ **Examples**: Mẫu dữ liệu thực tế
- ✅ **Field metadata**: Thông tin về từng trường
- ✅ **Viewer**: Hiển thị 20 dòng đầu tiên

## 📦 Cài đặt và Import

```bash
pip install datasets
```

```python
from datasets import load_dataset, load_dataset_builder
```

## 🔍 Kiểm tra Dataset trước khi tải

### Inspect metadata
```python
# Kiểm tra thông tin dataset trước khi download
builder = load_dataset_builder("imdb")

# Kiểm tra kích thước dataset
dataset_size_mb = builder.info.dataset_size / (1024 * 1024)
print(f"Dataset size: {dataset_size_mb:.1f} MB")

# Kiểm tra cấu trúc
print("Features:", builder.info.features)
print("Splits:", list(builder.info.splits.keys()))
```

### Thông tin quan trọng cần check:
- 📊 **Dataset size**: Đảm bảo đủ storage và bandwidth
- 🗂️ **Available splits**: train, test, validation
- 🏗️ **Data structure**: Các columns và data types
- 📝 **Description**: Hiểu rõ mục đích sử dụng

## 📥 Tải Dataset

### Tải toàn bộ dataset
```python
# Tải tất cả splits
dataset = load_dataset("imdb")
print(dataset)

# Kết quả:
# DatasetDict({
#     train: Dataset({features: ['text', 'label'], num_rows: 25000})
#     test: Dataset({features: ['text', 'label'], num_rows: 25000})
# })
```

### Tải split cụ thể
```python
# Chỉ tải train split
train_dataset = load_dataset("imdb", split="train")
print(f"Train samples: {len(train_dataset)}")

# Tải nhiều splits
test_dataset = load_dataset("imdb", split="test")
```

### Tải subset nhỏ để test
```python
# Tải chỉ 100 samples đầu tiên
small_dataset = load_dataset("imdb", split="train[:100]")
print(f"Small dataset: {len(small_dataset)} samples")
```

## 🛠️ Xử lý và Thao tác Dataset

### Apache Arrow Format
- 📊 **Columnar storage**: Lưu trữ theo cột thay vì hàng
- ⚡ **Hiệu suất cao**: Truy vấn và xử lý nhanh hơn
- 💾 **Memory efficient**: Tối ưu sử dụng RAM

### Filtering Data
```python
# Filter theo điều kiện
positive_reviews = dataset["train"].filter(lambda x: x["label"] == 1)
print(f"Positive reviews: {len(positive_reviews)}")

# Filter phức tạp hơn
long_texts = dataset["train"].filter(lambda x: len(x["text"]) > 500)
```

### Selecting Data
```python
# Lấy 2 rows đầu tiên
first_two = dataset["train"].select(range(2))
print(f"Selected: {len(first_two)} rows")

# Truy cập dữ liệu cụ thể
print("First text:", first_two[0]["text"])
print("First label:", first_two[0]["label"])
```

### Map và Transform
```python
# Áp dụng function cho tất cả rows
def preprocess_text(examples):
    return {"text": [text.lower().strip() for text in examples["text"]]}

processed_dataset = dataset["train"].map(preprocess_text, batched=True)
```

## 💡 Best Practices cho Developer Việt Nam

### 1. Kiểm tra trước khi tải
```python
# Luôn check size trước khi download
builder = load_dataset_builder("dataset_name")
size_gb = builder.info.dataset_size / (1024**3)
if size_gb > 5:  # Nếu > 5GB
    print(f"⚠️ Large dataset: {size_gb:.1f} GB")
    confirm = input("Continue? (y/n): ")
    if confirm.lower() != 'y':
        exit()
```

### 2. Sử dụng cache hiệu quả
```python
# Dataset được cache tự động tại ~/.cache/huggingface/datasets
# Để clear cache: rm -rf ~/.cache/huggingface/datasets
```

### 3. Working với dữ liệu lớn
```python
# Sử dụng streaming cho datasets khổng lồ
streaming_dataset = load_dataset("dataset_name", streaming=True)
for sample in streaming_dataset["train"].take(10):
    print(sample)
```

### 4. Lưu dataset đã xử lý
```python
# Lưu processed dataset
processed_dataset.save_to_disk("./my_processed_dataset")

# Load lại
reloaded_dataset = load_from_disk("./my_processed_dataset")
```

## 🎯 Lợi ích chính của Hugging Face Datasets

### ✅ Accessibility & Sharing
- 🌍 **Dễ truy cập**: Chỉ cần dataset path
- 🤝 **Dễ chia sẻ**: Tích hợp với Hugging Face ecosystem
- 📖 **Standardized**: Format thống nhất cho tất cả datasets

### ✅ Community Curation
- 👥 **Quality control**: Được cộng đồng review
- 🎯 **ML-ready**: Sẵn sàng cho các tác vụ ML phổ biến
- 📝 **Well-documented**: Có metadata và examples đầy đủ

### ✅ Performance
- ⚡ **Apache Arrow**: Xử lý nhanh và hiệu quả
- 💾 **Memory optimization**: Sử dụng RAM tối ưu
- 🔍 **Fast querying**: Truy vấn dữ liệu nhanh chóng

## 📋 Workflow Chuẩn

1. **🔍 Research**: Tìm dataset phù hợp trên Hub
2. **📊 Inspect**: Check metadata và size
3. **📥 Download**: Load dataset với split phù hợp
4. **👀 Explore**: Xem sample data và structure
5. **🛠️ Process**: Filter, map, transform theo nhu cầu
6. **💾 Save**: Lưu processed data nếu cần
7. **🚀 Use**: Integrate vào ML pipeline

## 🔗 Tài liệu tham khảo

- [Datasets Documentation](https://huggingface.co/docs/datasets/index)
- [Loading Datasets Guide](https://huggingface.co/docs/datasets/loading)
- [Processing Data](https://huggingface.co/docs/datasets/process)
- [Apache Arrow Overview](https://arrow.apache.org/overview/)

## 💡 Tips cho Developer Việt Nam

- 🌐 **Kết nối internet**: Đảm bảo stable khi download datasets lớn
- 🗂️ **Storage**: Kiểm tra dung lượng ổ cứng trước khi tải
- ⏰ **Time management**: Datasets lớn có thể mất nhiều thời gian download
- 🔄 **Caching**: Tận dụng cache để tránh download lại
- 📊 **Sample first**: Luôn test với subset nhỏ trước khi xử lý toàn bộ
