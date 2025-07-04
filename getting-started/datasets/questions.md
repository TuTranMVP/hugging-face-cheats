# Hugging Face Datasets Questions

## Question 7: exercise_7
**Title:** Dataset Inspection Best Practices

**Description:** Before loading large datasets from Hugging Face Hub, it's crucial to inspect their metadata to understand size, structure, and features. This helps avoid downloading unsuitable or oversized datasets.

**Question:** Which function should you use to inspect dataset metadata WITHOUT downloading the full dataset?

**Options:**
- load_dataset()
- load_dataset_builder()
- inspect_dataset()
- get_dataset_info()

**Correct Answer:** load_dataset_builder()

**Explanation:** Function `load_dataset_builder()` cho phép inspect metadata của dataset mà không cần download toàn bộ dữ liệu. Nó cung cấp thông tin về kích thước, features, splits, và cấu trúc dataset. Function `load_dataset()` sẽ download dataset thực tế, trong khi `inspect_dataset()` và `get_dataset_info()` không tồn tại trong thư viện datasets.

---

## Question 8: exercise_8
**Title:** Dataset Size Calculation

**Description:** When working with datasets, it's important to know their size to plan storage and bandwidth requirements. The dataset_size property returns size in bytes, which often needs conversion to more readable units.

**Question:** If a dataset builder shows `dataset_size = 52428800`, what is the correct way to convert this to megabytes?

**Options:**
- dataset_size / 1024
- dataset_size / (1024 ** 2)
- dataset_size / 1000000
- dataset_size * 1024

**Correct Answer:** dataset_size / (1024 ** 2)

**Explanation:** Để convert từ bytes sang megabytes, cần chia cho 1024^2 (1,048,576). Công thức `dataset_size / (1024 ** 2)` là cách chính xác để convert. Chia cho 1024 chỉ convert sang kilobytes, chia cho 1,000,000 sử dụng decimal system thay vì binary, và nhân với 1024 sẽ làm tăng kích thước thay vì giảm.

---

## Question 9: exercise_9
**Title:** Loading Dataset Splits

**Description:** Hugging Face datasets are typically divided into splits like train, test, and validation. You can load specific splits to save time and storage when you only need certain portions of the data.

**Question:** What is the correct syntax to load only the "test" split of the "TIGER-Lab/MMLU-Pro" dataset?

**Options:**
- load_dataset("TIGER-Lab/MMLU-Pro").test
- load_dataset("TIGER-Lab/MMLU-Pro", split="test")
- load_dataset("TIGER-Lab/MMLU-Pro", part="test")
- load_dataset("TIGER-Lab/MMLU-Pro", subset="test")

**Correct Answer:** load_dataset("TIGER-Lab/MMLU-Pro", split="test")

**Explanation:** Parameter `split="test"` là cách chính xác để load chỉ test split của dataset. Cú pháp này giúp tiết kiệm thời gian và storage bằng cách chỉ download phần dữ liệu cần thiết. Các options khác sử dụng sai tên parameter (`part`, `subset`) hoặc sai cú pháp (`.test`).

---

## Question 10: exercise_10
**Title:** Dataset Filtering with Apache Arrow

**Description:** Hugging Face datasets use Apache Arrow format, which requires specific methods for data manipulation. Filtering datasets is different from working with pandas DataFrames and uses lambda functions.

**Question:** Which method is used to filter a Hugging Face dataset for rows containing "football" in the text column?

**Options:**
- dataset.query('text.contains("football")')
- dataset.filter(lambda row: "football" in row["text"])
- dataset[dataset["text"].str.contains("football")]
- dataset.where(dataset["text"] == "football")

**Correct Answer:** dataset.filter(lambda row: "football" in row["text"])

**Explanation:** Method `.filter()` với lambda function là cách chính xác để filter Hugging Face datasets. Apache Arrow format yêu cầu sử dụng lambda functions thay vì pandas-style operations. Lambda function nhận từng row và return True/False để quyết định row có được giữ lại hay không. Các options khác sử dụng pandas syntax không tương thích với Arrow format.

---

## Question 11: exercise_11
**Title:** Dataset Selection and Slicing

**Description:** When working with large datasets, you often need to select specific rows or create smaller samples for testing and development. The select() method allows you to choose specific indices from a dataset.

**Question:** How do you select the first 5 rows from a filtered dataset using the select() method?

**Options:**
- filtered.select([0, 1, 2, 3, 4])
- filtered.select(range(5))
- filtered.head(5)
- filtered[:5]

**Correct Answer:** filtered.select(range(5))

**Explanation:** Method `.select(range(5))` là cách chính xác để select 5 rows đầu tiên từ dataset. Function `range(5)` tạo ra sequence [0, 1, 2, 3, 4] mà `.select()` method cần. Mặc dù `.select([0, 1, 2, 3, 4])` cũng hoạt động, nhưng `range(5)` ngắn gọn và dễ scale hơn. `.head()` và slice notation `[:5]` là pandas syntax không áp dụng cho Hugging Face datasets.

---

## Question 12: exercise_12
**Title:** Dataset Performance Considerations

**Description:** Working with large datasets requires understanding of performance implications. Dataset inspection, loading strategies, and manipulation methods all impact processing time and resource usage.

**Question:** Why is it recommended to use load_dataset_builder() before load_dataset() for large datasets?

**Options:**
- It's faster to load the actual data
- It allows you to check dataset size and structure without downloading
- It automatically optimizes the dataset format
- It provides better error handling

**Correct Answer:** It allows you to check dataset size and structure without downloading

**Explanation:** Function `load_dataset_builder()` cho phép inspect metadata (size, features, splits) mà không cần download dataset thực tế. Điều này đặc biệt quan trọng với large datasets có thể có kích thước GB hoặc TB. Bằng cách check trước, bạn có thể quyết định có nên download hay không, tránh waste bandwidth và storage. Nó không optimize format hay cung cấp better error handling so với `load_dataset()`.

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
3. **Đảm bảo exercise_id** là duy nhất (exercise_13, exercise_14, ...)
4. **Chạy lại tool** để test câu hỏi mới

## Lưu ý:
- Mỗi câu hỏi phải có **4 lựa chọn**
- **Correct Answer** phải khớp chính xác với một trong các Options
- **Explanation** nên bằng tiếng Việt để dễ hiểu
- Sử dụng `---` để phân cách giữa các câu hỏi

## Chủ đề đã cover:
- **Dataset Inspection**: load_dataset_builder(), metadata
- **Size Calculation**: bytes to MB conversion
- **Loading Splits**: split parameter usage
- **Filtering**: Apache Arrow format, lambda functions
- **Selection**: select() method, range() usage
- **Performance**: best practices cho large datasets
