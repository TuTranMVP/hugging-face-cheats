# Câu Hỏi về Hugging Face Datasets

## Câu Hỏi 7: exercise_7
**Tiêu đề:** Thực hành Tốt nhất cho Kiểm tra Dataset

**Mô tả:** Trước khi tải các dataset lớn từ Hugging Face Hub, điều quan trọng là phải kiểm tra metadata của chúng để hiểu kích thước, cấu trúc và các tính năng. Điều này giúp tránh tải xuống các dataset không phù hợp hoặc quá lớn.

**Câu hỏi:** Bạn nên sử dụng hàm nào để kiểm tra metadata của dataset MÀ KHÔNG tải xuống toàn bộ dataset?

**Các lựa chọn:**
- load_dataset()
- load_dataset_builder()
- inspect_dataset()
- get_dataset_info()

**Đáp án đúng:** load_dataset_builder()

**Giải thích:** Function `load_dataset_builder()` cho phép inspect metadata của dataset mà không cần download toàn bộ dữ liệu. Nó cung cấp thông tin về kích thước, features, splits, và cấu trúc dataset. Function `load_dataset()` sẽ download dataset thực tế, trong khi `inspect_dataset()` và `get_dataset_info()` không tồn tại trong thư viện datasets.

---

## Câu Hỏi 8: exercise_8
**Tiêu đề:** Tính toán Kích thước Dataset

**Mô tả:** Khi làm việc với datasets, điều quan trọng là phải biết kích thước của chúng để lên kế hoạch cho các yêu cầu lưu trữ và băng thông. Thuộc tính dataset_size trả về kích thước tính bằng byte, thường cần chuyển đổi thành các đơn vị dễ đọc hơn.

**Câu hỏi:** Nếu một dataset builder hiển thị `dataset_size = 52428800`, cách nào đúng để chuyển đổi sang megabytes?

**Các lựa chọn:**
- dataset_size / 1024
- dataset_size / (1024 ** 2)
- dataset_size / 1000000
- dataset_size * 1024

**Đáp án đúng:** dataset_size / (1024 ** 2)

**Giải thích:** Để convert từ bytes sang megabytes, cần chia cho 1024^2 (1,048,576). Công thức `dataset_size / (1024 ** 2)` là cách chính xác để convert. Chia cho 1024 chỉ convert sang kilobytes, chia cho 1,000,000 sử dụng decimal system thay vì binary, và nhân với 1024 sẽ làm tăng kích thước thay vì giảm.

---

## Câu Hỏi 9: exercise_9
**Tiêu đề:** Tải Dataset Splits

**Mô tả:** Datasets Hugging Face thường được chia thành các splits như train, test, và validation. Bạn có thể tải các splits cụ thể để tiết kiệm thời gian và dung lượng lưu trữ khi chỉ cần một số phần dữ liệu nhất định.

**Câu hỏi:** Cú pháp nào đúng để chỉ tải split "test" của dataset "TIGER-Lab/MMLU-Pro"?

**Các lựa chọn:**
- load_dataset("TIGER-Lab/MMLU-Pro").test
- load_dataset("TIGER-Lab/MMLU-Pro", split="test")
- load_dataset("TIGER-Lab/MMLU-Pro", part="test")
- load_dataset("TIGER-Lab/MMLU-Pro", subset="test")

**Đáp án đúng:** load_dataset("TIGER-Lab/MMLU-Pro", split="test")

**Giải thích:** Parameter `split="test"` là cách chính xác để load chỉ test split của dataset. Cú pháp này giúp tiết kiệm thời gian và storage bằng cách chỉ download phần dữ liệu cần thiết. Các options khác sử dụng sai tên parameter (`part`, `subset`) hoặc sai cú pháp (`.test`).

---

## Câu Hỏi 10: exercise_10
**Tiêu đề:** Lọc Dataset với Apache Arrow

**Mô tả:** Datasets Hugging Face sử dụng định dạng Apache Arrow, yêu cầu các phương thức cụ thể để thao tác dữ liệu. Việc lọc datasets khác với làm việc với pandas DataFrames và sử dụng lambda functions.

**Câu hỏi:** Phương thức nào được sử dụng để lọc dataset Hugging Face cho các hàng chứa "football" trong cột text?

**Các lựa chọn:**
- dataset.query('text.contains("football")')
- dataset.filter(lambda row: "football" in row["text"])
- dataset[dataset["text"].str.contains("football")]
- dataset.where(dataset["text"] == "football")

**Đáp án đúng:** dataset.filter(lambda row: "football" in row["text"])

**Giải thích:** Method `.filter()` với lambda function là cách chính xác để filter Hugging Face datasets. Apache Arrow format yêu cầu sử dụng lambda functions thay vì pandas-style operations. Lambda function nhận từng row và return True/False để quyết định row có được giữ lại hay không. Các options khác sử dụng pandas syntax không tương thích với Arrow format.

---

## Câu Hỏi 11: exercise_11
**Tiêu đề:** Lựa chọn và Cắt Dataset

**Mô tả:** Khi làm việc với các dataset lớn, bạn thường cần chọn các hàng cụ thể hoặc tạo các mẫu nhỏ hơn để kiểm tra và phát triển. Phương thức select() cho phép bạn chọn các chỉ mục cụ thể từ một dataset.

**Câu hỏi:** Làm thế nào để chọn 5 hàng đầu tiên từ một dataset đã được lọc bằng cách sử dụng phương thức select()?

**Các lựa chọn:**
- filtered.select([0, 1, 2, 3, 4])
- filtered.select(range(5))
- filtered.head(5)
- filtered[:5]

**Đáp án đúng:** filtered.select(range(5))

**Giải thích:** Method `.select(range(5))` là cách chính xác để select 5 rows đầu tiên từ dataset. Function `range(5)` tạo ra sequence [0, 1, 2, 3, 4] mà `.select()` method cần. Mặc dù `.select([0, 1, 2, 3, 4])` cũng hoạt động, nhưng `range(5)` ngắn gọn và dễ scale hơn. `.head()` và slice notation `[:5]` là pandas syntax không áp dụng cho Hugging Face datasets.

---

## Câu Hỏi 12: exercise_12
**Tiêu đề:** Xem xét Hiệu suất Dataset

**Mô tả:** Làm việc với các dataset lớn đòi hỏi hiểu biết về các tác động hiệu suất. Kiểm tra dataset, chiến lược tải và các phương thức thao tác đều ảnh hưởng đến thời gian xử lý và sử dụng tài nguyên.

**Câu hỏi:** Tại sao được khuyến nghị sử dụng load_dataset_builder() trước load_dataset() cho các dataset lớn?

**Các lựa chọn:**
- Nó nhanh hơn để tải dữ liệu thực tế
- Nó cho phép bạn kiểm tra kích thước và cấu trúc dataset mà không cần tải xuống
- Nó tự động tối ưu hóa định dạng dataset
- Nó cung cấp xử lý lỗi tốt hơn

**Đáp án đúng:** Nó cho phép bạn kiểm tra kích thước và cấu trúc dataset mà không cần tải xuống

**Giải thích:** Function `load_dataset_builder()` cho phép inspect metadata (size, features, splits) mà không cần download dataset thực tế. Điều này đặc biệt quan trọng với large datasets có thể có kích thước GB hoặc TB. Bằng cách check trước, bạn có thể quyết định có nên download hay không, tránh waste bandwidth và storage. Nó không optimize format hay cung cấp better error handling so với `load_dataset()`.

---

## Mẫu cho Câu hỏi Mới

```markdown
## Câu Hỏi X: exercise_X
**Tiêu đề:** [Tiêu đề câu hỏi]

**Mô tả:** [Mô tả ngữ cảnh của câu hỏi]

**Câu hỏi:** [Câu hỏi chính]

**Các lựa chọn:**
- [Lựa chọn 1]
- [Lựa chọn 2]
- [Lựa chọn 3]
- [Lựa chọn 4]

**Đáp án đúng:** [Đáp án đúng]

**Giải thích:** [Giải thích tại sao đáp án này đúng]

---
```

## Hướng dẫn thêm câu hỏi mới:

1. **Copy template** ở cuối file
2. **Thay thế các placeholder** [...] bằng nội dung thực tế
3. **Đảm bảo exercise_id** là duy nhất (exercise_13, exercise_14, ...)
4. **Chạy lại tool** để test câu hỏi mới

## Lưu ý:
- Mỗi câu hỏi phải có **4 lựa chọn**
- **Đáp án đúng** phải khớp chính xác với một trong các Lựa chọn
- **Giải thích** nên bằng tiếng Việt để dễ hiểu
- Sử dụng `---` để phân cách giữa các câu hỏi

## Chủ đề đã cover:
- **Kiểm tra Dataset**: load_dataset_builder(), metadata
- **Tính toán Kích thước**: chuyển đổi bytes sang MB
- **Tải Splits**: sử dụng parameter split
- **Lọc**: định dạng Apache Arrow, lambda functions
- **Lựa chọn**: phương thức select(), sử dụng range()
- **Hiệu suất**: thực hành tốt nhất cho large datasets
