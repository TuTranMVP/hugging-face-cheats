# ❓ Câu hỏi phỏng vấn: Auto Models và Tokenizers

## Question 1: auto_models_basic
**Title:** Hiểu biết cơ bản về Auto Classes

**Description:** Đánh giá kiến thức cơ bản về Auto Classes trong Hugging Face Transformers và khi nào nên sử dụng chúng thay vì Pipelines.

**Question:** Khi nào bạn nên sử dụng Auto Classes thay vì Pipelines trong Hugging Face Transformers?

**Options:**
- Khi cần prototype nhanh và không quan tâm đến tùy chỉnh
- Khi cần kiểm soát chi tiết quá trình preprocessing và có workflow phức tạp
- Khi mới bắt đầu học Hugging Face và cần approach đơn giản
- Khi chỉ cần kết quả demo nhanh chóng

**Correct Answer:** Khi cần kiểm soát chi tiết quá trình preprocessing và có workflow phức tạp

**Explanation:** Auto Classes cung cấp tính linh hoạt và kiểm soát cao hơn Pipelines, phù hợp cho production workflows, custom preprocessing, và khi cần tùy chỉnh thresholds. Pipelines tốt hơn cho rapid prototyping và demo đơn giản.

---

## Question 2: tokenizer_model_pairing
**Title:** Tầm quan trọng của việc ghép đôi Tokenizer và Model

**Description:** Kiểm tra hiểu biết về mối quan hệ giữa tokenizer và model trong Auto Classes.

**Question:** Tại sao việc sử dụng cùng model name cho cả AutoTokenizer và AutoModel là quan trọng?

**Options:**
- Để giảm thời gian download và tiết kiệm bandwidth
- Để đảm bảo tokenizer xử lý text giống như lúc model được training
- Để tương thích với các phiên bản Transformers library cũ hơn
- Để có thể sử dụng các tính năng cache của Hugging Face Hub

**Correct Answer:** Để đảm bảo tokenizer xử lý text giống như lúc model được training

**Explanation:** Tokenizer và model phải được ghép đôi để đảm bảo input được xử lý chính xác như trong quá trình training. Mỗi model được train với tokenizer cụ thể, việc mismatch có thể dẫn đến kết quả không chính xác.

---

## Question 3: automodel_classes
**Title:** Lựa chọn AutoModel Class phù hợp

**Description:** Đánh giá khả năng chọn đúng AutoModel class cho từng task cụ thể.

**Question:** AutoModel class nào phù hợp nhất cho task phân loại cảm xúc (sentiment analysis)?

**Options:**
- AutoModel - cho feature extraction chung
- AutoModelForSequenceClassification - cho text classification
- AutoModelForTokenClassification - cho token-level classification
- AutoModelForQuestionAnswering - cho reading comprehension

**Correct Answer:** AutoModelForSequenceClassification - cho text classification

**Explanation:** Sentiment analysis là một dạng sequence classification (phân loại toàn bộ sequence text), nên AutoModelForSequenceClassification là lựa chọn phù hợp nhất. Các class khác phục vụ cho tasks khác như token classification (NER) hoặc question answering.

---

## Question 4: tokenization_process
**Title:** Hiểu về quá trình Tokenization

**Description:** Kiểm tra hiểu biết về các bước trong quá trình tokenization.

**Question:** Thứ tự đúng của các bước trong quá trình tokenization là gì?

**Options:**
- Text cleaning → Add special tokens → Tokenization → Token to ID mapping
- Tokenization → Text cleaning → Token to ID mapping → Add special tokens
- Text cleaning → Tokenization → Token to ID mapping → Add special tokens
- Token to ID mapping → Tokenization → Text cleaning → Add special tokens

**Correct Answer:** Text cleaning → Tokenization → Token to ID mapping → Add special tokens

**Explanation:** Quá trình tokenization thường theo thứ tự: 1) Text cleaning (lowercase, remove accents), 2) Tokenization (chia thành tokens), 3) Token to ID mapping (convert tokens thành numbers), 4) Add special tokens ([CLS], [SEP]), cuối cùng là padding/truncation.

---

## Question 5: batch_processing
**Title:** Batch Processing với Auto Classes

**Description:** Đánh giá hiểu biết về batch processing để tối ưu performance.

**Question:** Khi xử lý nhiều texts cùng lúc với AutoTokenizer, parameter nào quan trọng nhất để đảm bảo consistency?

**Options:**
- max_length - để giới hạn độ dài sequence
- return_tensors - để return PyTorch hoặc TensorFlow tensors
- padding - để đảm bảo tất cả sequences có cùng length
- truncation - để cắt bớt sequences quá dài

**Correct Answer:** padding - để đảm bảo tất cả sequences có cùng length

**Explanation:** Khi batch processing, padding là quan trọng nhất để đảm bảo tất cả sequences trong batch có cùng length, cho phép model xử lý đồng thời. Các parameters khác cũng quan trọng nhưng padding là cốt lõi cho batch consistency.

---

## Question 6: performance_optimization
**Title:** Tối ưu Performance với Auto Classes

**Description:** Kiểm tra kiến thức về các techniques tối ưu performance khi sử dụng Auto Classes.

**Question:** Technique nào có impact lớn nhất để tăng tốc inference với Auto Classes?

**Options:**
- Sử dụng model caching để avoid reload model
- Batch processing nhiều inputs cùng lúc thay vì xử lý từng cái một
- Tăng max_length parameter để xử lý sequences dài hơn
- Sử dụng return_tensors="pt" thay vì default format

**Correct Answer:** Batch processing nhiều inputs cùng lúc thay vì xử lý từng cái một

**Explanation:** Batch processing có impact lớn nhất (3-5x faster) vì tận dụng được parallel computation của GPU/CPU. Model caching giúp nhưng chỉ khi reload model nhiều lần. Tăng max_length thực tế làm chậm, và return_tensors format không ảnh hưởng đáng kể đến speed.

---

## Question 7: custom_preprocessing
**Title:** Custom Preprocessing với Auto Classes

**Description:** Đánh giá khả năng tùy chỉnh preprocessing trong Auto Classes.

**Question:** Trong code sau, vấn đề gì có thể xảy ra?

```python
# Custom preprocessing
def custom_preprocess(text):
    return text.upper().replace("!", ".")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "Hello World!"
processed_text = custom_preprocess(text)
inputs = tokenizer(processed_text, return_tensors="pt")
```

**Options:**
- Code hoàn toàn chính xác và sẽ hoạt động tốt
- BERT uncased model không được train với uppercase text nên có thể cho kết quả không tốt
- Việc replace "!" thành "." sẽ gây lỗi tokenization
- return_tensors="pt" không tương thích với processed text

**Correct Answer:** BERT uncased model không được train với uppercase text nên có thể cho kết quả không tốt

**Explanation:** BERT-base-uncased được train với lowercase text, việc convert thành uppercase trong custom preprocessing có thể làm giảm accuracy vì mismatch với training data. Custom preprocessing cần phù hợp với cách model được train.

---

## Question 8: error_handling
**Title:** Xử lý lỗi phổ biến với Auto Classes

**Description:** Kiểm tra khả năng identify và fix các lỗi thường gặp.

**Question:** Lỗi gì có thể xảy ra với code sau và cách fix?

```python
texts = ["Very long text" * 1000, "Short text"]
inputs = tokenizer(texts, return_tensors="pt")
outputs = model(**inputs)
```

**Options:**
- Memory error do sequence quá dài - fix bằng truncation=True, max_length=512
- Tokenizer error do mixed length - fix bằng padding=True
- Model architecture error - fix bằng cách dùng đúng AutoModel class
- Tensor dimension error - fix bằng return_tensors="tf"

**Correct Answer:** Memory error do sequence quá dài - fix bằng truncation=True, max_length=512

**Explanation:** Text quá dài ("Very long text" * 1000) sẽ tạo ra sequence rất dài, có thể gây memory error. Fix bằng cách thêm truncation=True và max_length để giới hạn sequence length. Padding cũng nên thêm cho consistency.

---

## Question 9: model_outputs
**Title:** Hiểu về Model Outputs

**Description:** Kiểm tra hiểu biết về cách xử lý outputs từ Auto Models.

**Question:** Với AutoModelForSequenceClassification, làm thế nào để convert logits thành probabilities?

**Options:**
- Sử dụng torch.argmax() trực tiếp trên logits
- Apply torch.nn.functional.softmax() lên logits
- Sử dụng torch.sigmoid() cho multi-class classification
- Chia logits cho tổng của chúng

**Correct Answer:** Apply torch.nn.functional.softmax() lên logits

**Explanation:** Logits là raw outputs từ model, cần apply softmax để convert thành probabilities (sum to 1). torch.argmax() chỉ cho predicted class, không phải probabilities. Sigmoid dùng cho binary/multi-label, không phải multi-class.

---

## Question 10: production_considerations
**Title:** Production Deployment Considerations

**Description:** Đánh giá hiểu biết về các yếu tố cần xem xét khi deploy Auto Classes lên production.

**Question:** Khi deploy model với Auto Classes lên production, practice nào KHÔNG được khuyến nghị?

**Options:**
- Set model.eval() để disable dropout và batch normalization
- Sử dụng torch.no_grad() context để giảm memory usage
- Load model mới cho mỗi request để đảm bảo consistency
- Implement error handling cho invalid inputs

**Correct Answer:** Load model mới cho mỗi request để đảm bảo consistency

**Explanation:** Load model cho mỗi request là rất kém hiệu quả và không cần thiết. Model nên được load một lần khi khởi động service và reuse cho tất cả requests. Các practices khác đều là best practices cho production deployment.

---

## Question 11: tokenizer_methods
**Title:** Hiểu về các methods của AutoTokenizer

**Description:** Đánh giá kiến thức về các methods khác nhau của AutoTokenizer và cách sử dụng chúng.

**Question:** Trong AutoTokenizer, sự khác biệt chính giữa method `.tokenize()` và `.encode()` là gì?

**Options:**
- `.tokenize()` trả về tokens dạng string, `.encode()` trả về token IDs dạng numbers
- `.tokenize()` nhanh hơn `.encode()` trong việc xử lý
- `.tokenize()` hỗ trợ batch processing, `.encode()` chỉ xử lý single text
- `.tokenize()` thêm special tokens, `.encode()` không thêm

**Correct Answer:** `.tokenize()` trả về tokens dạng string, `.encode()` trả về token IDs dạng numbers

**Explanation:** `.tokenize()` chia text thành tokens dạng string (human-readable), trong khi `.encode()` convert text thành token IDs dạng numbers mà model có thể xử lý. `.encode()` thường được sử dụng nhiều hơn vì model cần input dạng numbers.

---

## Question 12: text_preparation_process
**Title:** Quy trình chuẩn bị text với AutoTokenizer

**Description:** Kiểm tra hiểu biết về các bước xử lý text trong AutoTokenizer.

**Question:** AutoTokenizer thực hiện những bước nào trong quá trình text preparation?

**Options:**
- Chỉ tokenization - chia text thành từng phần nhỏ
- Chỉ cleaning và normalization - chuẩn hóa text
- Cleaning, normalization, và tokenization - tất cả các bước xử lý
- Chỉ convert text thành numbers cho model

**Correct Answer:** Cleaning, normalization, và tokenization - tất cả các bước xử lý

**Explanation:** AutoTokenizer tự động thực hiện toàn bộ quá trình: cleaning (loại bỏ ký tự không cần thiết), normalization (chuẩn hóa như lowercase), và tokenization (chia thành tokens). Đây là lý do AutoTokenizer rất tiện lợi cho developers.

---

## Question 13: model_checkpoint_switching
**Title:** Chuyển đổi models trong rapid experimentation

**Description:** Đánh giá hiểu biết về việc so sánh models nhanh chóng.

**Question:** Khi muốn so sánh hiệu suất của nhiều models khác nhau cho cùng một task, approach nào hiệu quả nhất?

**Options:**
- Viết lại code từ đầu cho mỗi model để đảm bảo accuracy
- Sử dụng Pipeline và chỉ thay đổi model checkpoint name
- Sử dụng Auto Classes riêng biệt cho từng model
- Download tất cả models trước rồi so sánh offline

**Correct Answer:** Sử dụng Pipeline và chỉ thay đổi model checkpoint name

**Explanation:** Pipeline cho phép chuyển đổi models chỉ bằng cách thay model name, rất hiệu quả cho rapid experimentation. Code không cần thay đổi, chỉ cần update model checkpoint, giúp so sánh nhanh chóng và nhất quán.

---

## Question 14: balance_control_convenience
**Title:** Cân bằng giữa control và convenience

**Description:** Kiểm tra hiểu biết về việc kết hợp AutoClasses với Pipeline.

**Question:** Khi nào việc kết hợp AutoModel, AutoTokenizer với pipeline() function là lựa chọn tối ưu?

**Options:**
- Khi cần control hoàn toàn và không quan tâm đến convenience
- Khi muốn convenience hoàn toàn và không cần customization
- Khi cần balance giữa control và convenience - có thể customize components nhưng vẫn dễ sử dụng
- Khi đang học và muốn hiểu cách hoạt động của từng component

**Correct Answer:** Khi cần balance giữa control và convenience - có thể customize components nhưng vẫn dễ sử dụng

**Explanation:** Kết hợp AutoClasses với pipeline() cho phép customize model và tokenizer (control) nhưng vẫn sử dụng pipeline interface dễ dàng (convenience). Đây là approach tốt cho production khi cần vừa flexibility vừa ease of use.

---

## Question 15: specialized_tokenization
**Title:** Tokenization cho specialized domains

**Description:** Đánh giá khả năng apply AutoTokenizer cho các domain chuyên biệt.

**Question:** Khi xử lý văn bản tài chính có chứa các thuật ngữ như "EBITDA", "P/E ratio", làm thế nào để đảm bảo AutoTokenizer xử lý đúng?

**Options:**
- Sử dụng tokenizer mặc định, nó sẽ tự động học các thuật ngữ mới
- Chọn tokenizer đã được train trên financial data hoặc fine-tune tokenizer hiện có
- Convert tất cả thuật ngữ tài chính thành từ thông thường trước khi tokenize
- Loại bỏ các thuật ngữ phức tạp để tránh lỗi tokenization

**Correct Answer:** Chọn tokenizer đã được train trên financial data hoặc fine-tune tokenizer hiện có

**Explanation:** Để xử lý đúng specialized terms, cần tokenizer hiểu domain đó. Có thể sử dụng pre-trained tokenizer cho financial domain hoặc fine-tune tokenizer hiện có với financial vocabulary. Tokenizer mặc định có thể không hiểu specialized terms đúng cách.

---

## Question 16: practical_implementation
**Title:** Thực hành implementation trong real scenarios

**Description:** Kiểm tra khả năng implement AutoClasses trong scenarios thực tế.

**Question:** Trong code sau, vấn đề gì có thể xảy ra và cách fix?

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
```

**Options:**
- Code hoàn toàn đúng và sẽ hoạt động tốt
- Tokenizer và model không match - nên sử dụng cùng model name
- Pipeline function bị import thiếu - cần import từ transformers
- Model class không phù hợp với task sentiment analysis

**Correct Answer:** Tokenizer và model không match - nên sử dụng cùng model name

**Explanation:** BERT và DistilBERT có tokenizers khác nhau, việc mix có thể gây inconsistency. Nên sử dụng cùng model name cho cả tokenizer và model để đảm bảo compatibility và accuracy tốt nhất.

---

## Question 17: rapid_experimentation_workflow
**Title:** Workflow cho rapid experimentation

**Description:** Đánh giá hiểu biết về best practices cho việc thử nghiệm nhanh.

**Question:** Workflow nào tốt nhất để rapidly experiment với multiple models cho một task?

**Options:**
- Download tất cả models trước → Test từng cái một → So sánh kết quả
- Sử dụng Pipeline với list of model names → Loop qua từng model → Collect metrics
- Viết separate scripts cho mỗi model → Chạy parallel → Merge results
- Sử dụng AutoClasses riêng với manual comparison cho mỗi model

**Correct Answer:** Sử dụng Pipeline với list of model names → Loop qua từng model → Collect metrics

**Explanation:** Pipeline approach với loop qua model names là efficient nhất cho rapid experimentation. Code đơn giản, consistent, và dễ collect metrics để so sánh. Approach này tận dụng được convenience của Pipeline để rapid testing.

---

## Question 18: tokenizer_output_understanding
**Title:** Hiểu về output của tokenizer

**Description:** Kiểm tra hiểu biết về cách interpret output từ tokenizer methods.

**Question:** Khi gọi `tokenizer.tokenize("AI: Making robots smarter!")`, output có thể là gì?

**Options:**
- `["AI", ":", "Making", "robots", "smarter", "!"]` - exact word splitting
- `["ai", ":", "making", "robots", "smart", "##er", "!"]` - subword tokenization
- `[101, 7270, 1024, 2107, 8403, 6070, 999, 102]` - token IDs
- `{"input_ids": [...], "attention_mask": [...]}` - dictionary format

**Correct Answer:** `["ai", ":", "making", "robots", "smart", "##er", "!"]` - subword tokenization

**Explanation:** `.tokenize()` method trả về tokens dạng strings, often với subword tokenization (như "##er" cho suffix). Token IDs là output của `.encode()`, và dictionary format là của `.encode()` với `return_tensors` parameter.

---

## Question 19: custom_pipeline_creation
**Title:** Tạo custom pipeline với AutoClasses

**Description:** Đánh giá khả năng tạo pipeline tùy chỉnh hiệu quả.

**Question:** Khi tạo custom pipeline bằng cách combine AutoModel và AutoTokenizer, step nào QUAN TRỌNG NHẤT?

**Options:**
- Đảm bảo model và tokenizer sử dụng cùng checkpoint name
- Set correct device (CPU/GPU) cho model
- Specify đúng task type trong pipeline function
- Configure max_length và padding parameters

**Correct Answer:** Đảm bảo model và tokenizer sử dụng cùng checkpoint name

**Explanation:** Consistency giữa model và tokenizer là critical nhất. Nếu mismatch, sẽ có input processing errors và poor results. Device setting, task type, và parameters quan trọng nhưng không critical như model-tokenizer pairing.

---

## Question 20: production_vs_experimentation
**Title:** Production deployment vs Experimentation

**Description:** Kiểm tra hiểu biết về khi nào dùng approach nào.

**Question:** Khi deploy model lên production so với experimentation phase, strategy nào khác nhau?

**Options:**
- Production: Sử dụng Pipeline cho simplicity; Experimentation: AutoClasses cho flexibility
- Production: AutoClasses cho control; Experimentation: Pipeline cho rapid testing
- Production và Experimentation đều nên sử dụng same approach để consistency
- Production: Manual implementation; Experimentation: Sử dụng pre-built solutions

**Correct Answer:** Production: AutoClasses cho control; Experimentation: Pipeline cho rapid testing

**Explanation:** Experimentation phase cần rapid testing và comparison, nên Pipeline hiệu quả hơn. Production cần fine-tuned control, error handling, và optimization, nên AutoClasses phù hợp hơn. Mỗi phase có requirements khác nhau.

---
