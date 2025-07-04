"""
🎯 Mục tiêu bài tập:
Xây dựng text generation pipeline sử dụng model GPT-2
Tùy chỉnh output bằng cách điều chỉnh các tham số
Thử nghiệm với các prompt khác nhau

✅ Yêu cầu cụ thể:
Import pipeline từ transformers
Tạo pipeline với task "text-generation" và model "gpt2"
Cấu hình pipeline để tạo tối đa 10 tokens và 3 outputs
Cung cấp prompt tùy chỉnh làm input
Diễn giải cho lập trình viên Việt Nam:
Text Generation Pipeline là một công cụ mạnh mẽ cho phép:

🤖 Tự động tạo văn bản từ một câu khởi đầu (prompt)
⚙️ Điều chỉnh độ dài output thông qua max_length
🔢 Tạo nhiều phiên bản khác nhau với num_return_sequences
🎯 Sử dụng đơn giản chỉ với vài dòng code


🔑 Các giá trị đã điền:
pipeline - Import function từ transformers
pipeline - Tạo pipeline object
"gpt2" - Tên model GPT-2
"What if artificial intelligence could" - Prompt sáng tạo
10 - max_length = 10 tokens
3 - num_return_sequences = 3 outputs

💡 Đặc điểm nổi bật:
Prompt hay: "What if artificial intelligence could" - kích thích sự sáng tạo
Tham số hợp lý: 10 tokens và 3 variations
Code sạch: Tuân thủ PEP8 và best practices
Demo mở rộng: Thử nghiệm với nhiều prompts khác nhau
"""

from transformers.pipelines import pipeline  # type: ignore[import-untyped]

my_pipeline = pipeline(task='text-generation', model='gpt2')

# Tạo ba đầu ra văn bản với độ dài tối đa 10 mã thông báo
results = my_pipeline(
    'What if artificial intelligence could', max_length=10, num_return_sequences=3
)

# Display each result
for result in results:
    print(result['generated_text'])
