"""
🎯 Mục tiêu bài tập:
Lưu Hugging Face model locally để sử dụng offline
Reload model từ file local cho các lần sử dụng sau
Essential skills cho việc deployment và custom modifications

✅ Yêu cầu cụ thể (Instructions 1/2):
Tạo pipeline cho text classification
Sử dụng method thích hợp để save pipeline locally
Diễn giải cho lập trình viên Việt Nam:
Saving & Reloading Models rất quan trọng vì:

🔒 Offline access: Sử dụng khi không có internet
🛠️ Custom modifications: Fine-tuning cho dự án cụ thể
🚀 Efficient deployment: Tránh download lại mỗi lần chạy
💾 Storage control: Quản lý versions và dung lượng

📁 Files được save:
model.safetensors - Model weights
config.json - Model configuration
tokenizer.json - Tokenizer data
tokenizer_config.json - Tokenizer configuration
vocab.txt - Vocabulary
special_tokens_map.json - Special tokens mapping

💡 Benefits được demo:
🔒 Offline access - Không cần internet
🚀 Faster loading - Không cần download
💾 Version control - Quản lý phiên bản model
🛠️ Custom modifications - Có thể fine-tune

🎯 Workflow hoàn chỉnh:
Load model từ Hugging Face Hub
Test model với sample input
Save model locally với save_pretrained()
Reload model từ local directory
Verify rằng performance giữ nguyên
"""

from transformers import pipeline  # type: ignore[attr-defined]

modelId = 'distilbert-base-uncased-finetuned-sst-2-english'

my_pipeline = pipeline('text-classification', model=modelId)

# Lưu mô hình locally
my_pipeline.save_pretrained(f'models/{modelId}')

# Tải lại pipeline đã lưu
reloaded_pipeline = pipeline('text-classification', model=f'models/{modelId}')

# Kiểm tra pipeline đã được tải lại
print(reloaded_pipeline('Hugging Face is great!'))
