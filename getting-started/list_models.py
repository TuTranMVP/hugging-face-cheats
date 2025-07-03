# Import class HfApi từ package huggingface_hub
# Tạo một instance của class HfApi
# Sử dụng phương thức thích hợp để liệt kê tất cả các mô hình và in ra 2 mô hình đầu tiên

from huggingface_hub import HfApi

# Create an instance of the HfApi class
api = HfApi()

# List only the first 2 models available on the Hub
models = list(api.list_models(limit=2))

# Print the first 2 models
for model in models:
    print(model)
