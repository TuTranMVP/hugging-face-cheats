"""
Text Generation Pipeline với GPT-2
Bài tập: Xây dựng pipeline tạo văn bản tự động với các tham số tùy chỉnh
"""

from transformers import pipeline

# Tạo text generation pipeline với model GPT-2
my_pipeline = pipeline(task='text-generation', model='gpt2')

# Tạo 3 văn bản với độ dài tối đa 10 tokens
results = my_pipeline(
    'What if artificial intelligence could', max_length=10, num_return_sequences=3
)

# Hiển thị từng kết quả
print('🤖 Text Generation Results:')
print('=' * 50)

for i, result in enumerate(results, 1):
    print(f'Output {i}: {result["generated_text"]}')
    print('-' * 30)

# Thử nghiệm với các prompt khác
print('\n🎯 Experimenting with different prompts:')
print('=' * 50)

# Prompt về công nghệ
tech_results = my_pipeline(
    'How to become a better programmer', max_length=15, num_return_sequences=2
)
print('\n📱 Tech Prompt Results:')
for i, result in enumerate(tech_results, 1):
    print(f'{i}. {result["generated_text"]}')

# Prompt sáng tạo
creative_results = my_pipeline(
    'In a world where', max_length=12, num_return_sequences=2
)
print('\n✨ Creative Prompt Results:')
for i, result in enumerate(creative_results, 1):
    print(f'{i}. {result["generated_text"]}')

print('\n💡 Tips:')
print('- max_length: Độ dài tối đa của văn bản được tạo')
print('- num_return_sequences: Số lượng phiên bản khác nhau được tạo')
print('- Thử nghiệm với các prompt khác nhau để có kết quả thú vị hơn!')
