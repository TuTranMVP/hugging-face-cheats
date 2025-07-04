"""
Saving and Reloading Hugging Face Models
Bài tập: Lưu model locally và reload để sử dụng lại
"""

import os

from transformers import pipeline  # type: ignore[attr-defined]

# Model ID cho text classification
modelId = 'distilbert-base-uncased-finetuned-sst-2-english'

# Tạo pipeline cho text classification
my_pipeline = pipeline('text-classification', model=modelId)

# Tạo thư mục models nếu chưa tồn tại
os.makedirs('models', exist_ok=True)

# Save the model locally
print(f'💾 Saving model {modelId} locally...')
my_pipeline.save_pretrained(f'models/{modelId}')
print(f'✅ Model saved to: models/{modelId}')

# Test pipeline trước khi save
print('\n🔍 Testing original pipeline:')
test_text = 'I love this product!'
result = my_pipeline(test_text)
print(f"Text: '{test_text}'")
print(f'Result: {result[0]["label"]} (confidence: {result[0]["score"]:.3f})')

# Reload model từ local (Part 2 của exercise)
print('\n📁 Reloading model from local directory...')
local_pipeline = pipeline('text-classification', model=f'models/{modelId}')
print('✅ Model reloaded successfully!')

# Test reloaded pipeline
print('\n🔍 Testing reloaded pipeline:')
result_local = local_pipeline(test_text)
print(f"Text: '{test_text}'")
print(
    f'Result: {result_local[0]["label"]} (confidence: {result_local[0]["score"]:.3f})'
)

# Kiểm tra kích thước model đã save
model_path = f'models/{modelId}'
if os.path.exists(model_path):
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(model_path)
        for filename in filenames
    )
    print(f'\n📊 Model size: {total_size / (1024 * 1024):.1f} MB')

    # Liệt kê files đã save
    print(f'\n📁 Files saved in {model_path}:')
    for file in os.listdir(model_path):
        print(f'  - {file}')

print('\n💡 Benefits of saving models locally:')
print('  🔒 Offline access - No internet required')
print('  🚀 Faster loading - No download time')
print('  💾 Version control - Keep specific model versions')
print('  🛠️ Custom modifications - Fine-tune for your use case')
