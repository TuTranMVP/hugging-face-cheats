# ❓ Câu hỏi phỏng vấn: Hugging Face Complete Course

## Question 1: huggingface_hub_overview
**Title:** Hiểu biết về Hugging Face Hub

**Description:** Đánh giá kiến thức cơ bản về Hugging Face Hub và các tính năng chính của nó.

**Question:** Hugging Face Hub là gì và tính năng chính nào làm cho nó trở thành central platform cho AI community?

**Options:**
- Chỉ là nơi lưu trữ code và không có tính năng đặc biệt nào
- Central platform để discover, share models và datasets với model cards chi tiết
- Chỉ hỗ trợ GPT models và không support các loại models khác
- Tool để training models từ scratch mà không cần pre-trained models

**Correct Answer:** Central platform để discover, share models và datasets với model cards chi tiết

**Explanation:** Hugging Face Hub là nền tảng trung tâm cho AI community, cung cấp thousands of pre-trained models, datasets, và detailed documentation thông qua model cards. Nó hỗ trợ collaboration và sharing trong AI community.

---

## Question 2: pipelines_vs_auto_classes
**Title:** So sánh Pipelines và Auto Classes

**Description:** Kiểm tra hiểu biết về sự khác biệt và khi nào sử dụng Pipelines vs Auto Classes.

**Question:** Khi nào bạn nên chọn Auto Classes thay vì Pipelines trong Hugging Face?

**Options:**
- Khi cần rapid prototyping và không quan tâm đến customization
- Khi cần full control, custom preprocessing, và advanced workflows
- Khi mới bắt đầu học Hugging Face và cần simplicity
- Khi chỉ cần quick demo và không deploy production

**Correct Answer:** Khi cần full control, custom preprocessing, và advanced workflows

**Explanation:** Auto Classes cung cấp fine-grained control, suitable cho production applications, custom preprocessing, và advanced workflows. Pipelines tốt hơn cho rapid prototyping và simple use cases.

---

## Question 3: text_classification_applications
**Title:** Ứng dụng Text Classification

**Description:** Đánh giá hiểu biết về các use cases thực tế của text classification.

**Question:** Trong các scenario sau, đâu KHÔNG phải là use case phù hợp cho text classification?

**Options:**
- Phân loại email thành spam hoặc không spam
- Phân tích cảm xúc của customer reviews
- Tóm tắt một báo cáo tài chính dài thành version ngắn gọn
- Categorize tin tức thành các chủ đề như sports, politics, technology

**Correct Answer:** Tóm tắt một báo cáo tài chính dài thành version ngắn gọn

**Explanation:** Tóm tắt văn bản là task của text summarization, không phải text classification. Text classification gán labels/categories cho text, như spam detection, sentiment analysis, và topic categorization.

---

## Question 4: summarization_parameters
**Title:** Tùy chỉnh Summarization Parameters

**Description:** Kiểm tra hiểu biết về cách control output của summarization models.

**Question:** Khi sử dụng summarization pipeline, parameter nào quan trọng nhất để control độ dài của summary output?

**Options:**
- temperature - để control creativity của summary
- do_sample - để enable/disable random sampling
- max_length và min_length - để set giới hạn độ dài summary
- num_beams - để improve quality thông qua beam search

**Correct Answer:** max_length và min_length - để set giới hạn độ dài summary

**Explanation:** max_length và min_length directly control độ dài của summary output. Temperature và do_sample affect randomness, num_beams affects quality, nhưng length parameters là key để control output size.

---

## Question 5: document_qa_workflow
**Title:** Document Q&A Implementation Workflow

**Description:** Đánh giá hiểu biết về quy trình implement document question answering.

**Question:** Trong Document Q&A workflow, thứ tự đúng của các bước là gì?

**Options:**
- Extract text → Load Q&A model → Process question → Combine pages
- Load PDF → Extract pages → Combine text → Q&A pipeline → Answer
- Q&A pipeline → Load PDF → Extract text → Process answer
- Process question → Load PDF → Extract answer → Combine results

**Correct Answer:** Load PDF → Extract pages → Combine text → Q&A pipeline → Answer

**Explanation:** Workflow đúng: 1) Load PDF với PyPDF, 2) Extract text từ từng page, 3) Combine thành single text, 4) Pass question + context đến Q&A pipeline, 5) Get answer với confidence score.

---

## Question 6: pypdf_text_extraction
**Title:** PyPDF Text Extraction

**Description:** Kiểm tra kiến thức về cách sử dụng PyPDF để extract text từ PDF files.

**Question:** Code nào đúng để extract text từ tất cả pages của PDF file bằng PyPDF?

**Options:**
- `reader = PdfReader(file); text = reader.extract_all_text()`
- `reader = PdfReader(file); text = "".join([p.extract_text() for p in reader.pages])`
- `reader = PdfReader(file); text = reader.pages.extract_text()`
- `reader = PdfReader(file); text = reader.get_all_text()`

**Correct Answer:** `reader = PdfReader(file); text = "".join([p.extract_text() for p in reader.pages])`

**Explanation:** PyPDF requires iterate qua từng page trong reader.pages và call extract_text() method cho mỗi page, sau đó combine results. Không có built-in method để extract all text at once.

---

## Question 7: qa_pipeline_components
**Title:** Q&A Pipeline Components

**Description:** Đánh giá hiểu biết về các components cần thiết cho Q&A pipeline.

**Question:** Q&A pipeline cần những inputs nào để function correctly?

**Options:**
- Chỉ cần question string
- Question và context document text
- Question, context, và expected answer
- Context document và confidence threshold

**Correct Answer:** Question và context document text

**Explanation:** Q&A pipeline requires question (string to ask) và context (document text to search trong). Expected answer không cần thiết cho inference, và confidence threshold optional.

---

## Question 8: model_saving_loading
**Title:** Model Management trong Hugging Face

**Description:** Kiểm tra hiểu biết về cách save và load models locally.

**Question:** Method nào được sử dụng để save pre-trained model locally trong Hugging Face?

**Options:**
- `.save_model()` - standard saving method
- `.save_pretrained()` - Hugging Face specific method
- `.export_model()` - export cho production
- `.serialize()` - general serialization method

**Correct Answer:** `.save_pretrained()` - Hugging Face specific method

**Explanation:** `.save_pretrained()` là method chính trong Hugging Face để save models locally, tương ứng với `.from_pretrained()` để load. Nó saves both model weights và configuration.

---

## Question 9: dataset_preprocessing
**Title:** Dataset Preprocessing với Hugging Face

**Description:** Đánh giá kiến thức về dataset manipulation functions.

**Question:** Để chọn 1000 samples đầu tiên từ Hugging Face dataset, method nào đúng?

**Options:**
- `dataset.head(1000)` - pandas-style selection
- `dataset.select(range(1000))` - Hugging Face specific
- `dataset.slice(0, 1000)` - standard slicing
- `dataset.take(1000)` - TensorFlow style

**Correct Answer:** `dataset.select(range(1000))` - Hugging Face specific

**Explanation:** Hugging Face datasets sử dụng `.select()` method với indices để select specific samples. `.filter()` cho conditional selection, `.map()` cho transformations.

---

## Question 10: production_considerations
**Title:** Production Deployment Considerations

**Description:** Kiểm tra hiểu biết về factors quan trọng khi deploy models lên production.

**Question:** Khi deploy Hugging Face models lên production, factor nào KHÔNG quan trọng?

**Options:**
- Error handling và graceful failures
- Model performance monitoring và alerting
- Exact reproduction của training environment
- Security measures cho data privacy

**Correct Answer:** Exact reproduction của training environment

**Explanation:** Production deployment focus vào reliability, monitoring, security. Exact training environment reproduction không cần thiết vì pre-trained models đã stable. Error handling, monitoring, và security là critical cho production systems.

---

## Question 11: advanced_qa_features
**Title:** Advanced Q&A Features

**Description:** Đánh giá hiểu biết về advanced features của Q&A pipelines.

**Question:** Trong Q&A pipeline, parameter `handle_impossible_answer=True` có tác dụng gì?

**Options:**
- Tự động generate answer khi không tìm thấy trong context
- Allow model trả lời "không tìm thấy" thay vì force incorrect answer
- Improve performance bằng cách skip difficult questions
- Enable multi-language answer generation

**Correct Answer:** Allow model trả lời "không tìm thấy" thay vì force incorrect answer

**Explanation:** `handle_impossible_answer=True` cho phép model nhận biết khi answer không có trong context và trả về appropriate response thay vì force guess, improving reliability.

---

## Question 12: summarization_types
**Title:** Types of Summarization

**Description:** Kiểm tra hiểu biết về extractive vs abstractive summarization.

**Question:** Sự khác biệt chính giữa extractive và abstractive summarization là gì?

**Options:**
- Extractive nhanh hơn, abstractive chậm hơn
- Extractive chọn sentences từ text gốc, abstractive tạo new language
- Extractive cho short texts, abstractive cho long texts
- Extractive require less memory, abstractive require more

**Correct Answer:** Extractive chọn sentences từ text gốc, abstractive tạo new language

**Explanation:** Extractive summarization picks important sentences trực tiếp từ original text, trong khi abstractive summarization generates new sentences để express main ideas, similar to human-written summaries.

---

## Question 13: tokenizer_preprocessing
**Title:** Tokenizer Preprocessing Steps

**Description:** Đánh giá hiểu biết về quá trình tokenization trong Auto Classes.

**Question:** Khi sử dụng AutoTokenizer, parameter nào đảm bảo inputs có same length cho batch processing?

**Options:**
- truncation=True - cắt sequences dài
- return_tensors="pt" - return PyTorch tensors
- padding=True - pad sequences thành same length
- add_special_tokens=True - thêm [CLS], [SEP] tokens

**Correct Answer:** padding=True - pad sequences thành same length

**Explanation:** `padding=True` ensures tất cả sequences trong batch có same length bằng cách add padding tokens, essential cho batch processing. Truncation cắt long sequences, nhưng padding handles length consistency.

---

## Question 14: real_world_applications
**Title:** Real-world Applications Integration

**Description:** Kiểm tra khả năng apply Hugging Face vào business scenarios.

**Question:** Một company muốn automate customer support bằng cách answer questions từ product manuals. Approach tốt nhất là gì?

**Options:**
- Sử dụng text classification để categorize questions
- Sử dụng summarization để tóm tắt manuals
- Sử dụng document Q&A với manual text làm context
- Sử dụng text generation để create new answers

**Correct Answer:** Sử dụng document Q&A với manual text làm context

**Explanation:** Document Q&A perfect cho use case này: extract text từ product manuals (PyPDF), sử dụng Q&A pipeline để find specific answers cho customer questions. Classification hoặc summarization không solve direct question answering need.

---

## Question 15: performance_optimization
**Title:** Performance Optimization Strategies

**Description:** Đánh giá hiểu biết về cách optimize Hugging Face models cho production.

**Question:** Để optimize inference speed cho production deployment, strategy nào KHÔNG effective?

**Options:**
- Sử dụng smaller models như DistilBERT thay vì BERT
- Implement caching cho frequently asked questions
- Increase max_length parameter để process longer sequences
- Use GPU acceleration với appropriate device settings

**Correct Answer:** Increase max_length parameter để process longer sequences

**Explanation:** Increasing max_length actually slows down inference vì more computation required. Effective optimizations include smaller models (DistilBERT), caching, GPU acceleration, và limiting sequence length when possible.

---

## Question 16: error_handling_best_practices
**Title:** Error Handling Best Practices

**Description:** Kiểm tra hiểu biết về robust error handling trong production systems.

**Question:** Trong production Q&A system, cách handle errors tốt nhất khi PDF extraction fails là gì?

**Options:**
- Return empty string và continue processing
- Raise exception và stop entire application
- Log error, return meaningful message, và continue với other requests
- Retry extraction với different library

**Correct Answer:** Log error, return meaningful message, và continue với other requests

**Explanation:** Production systems cần robust error handling: log errors cho debugging, return user-friendly messages, và maintain service availability cho other requests. Stopping entire application hoặc silent failures không acceptable trong production.

---

## Question 17: model_selection_criteria
**Title:** Model Selection Criteria

**Description:** Đánh giá khả năng chọn model phù hợp cho specific use cases.

**Question:** Khi chọn model cho document Q&A system, factors nào quan trọng nhất?

**Options:**
- Model size và inference speed only
- Accuracy trên domain-specific data và response quality
- Number of downloads trên Hugging Face Hub
- Release date của model (newer is always better)

**Correct Answer:** Accuracy trên domain-specific data và response quality

**Explanation:** Model selection should prioritize performance trên actual use case data và quality of responses. Size/speed quan trọng nhưng accuracy critical. Downloads và release date không necessarily indicate suitability cho specific domain.

---

## Question 18: multi_document_qa
**Title:** Multi-document Q&A Challenges

**Description:** Kiểm tra hiểu biết về challenges khi scale Q&A lên multiple documents.

**Question:** Khi xử lý multiple PDF documents trong Q&A system, challenge chính là gì?

**Options:**
- PDF format compatibility issues
- Context length limitations của models
- Memory usage cho large document sets
- All of the above

**Correct Answer:** All of the above

**Explanation:** Multi-document Q&A faces multiple challenges: PDF compatibility issues, model context length limits (typically 512-1024 tokens), và memory consumption cho large document collections. Requires careful system design để handle these constraints.

---

## Question 19: continuous_learning
**Title:** Continuous Learning và Model Updates

**Description:** Đánh giá hiểu biết về maintaining models trong production environment.

**Question:** Để keep Q&A system updated với new information, best practice là gì?

**Options:**
- Retrain model from scratch mỗi tháng
- Update document collection và monitor performance metrics
- Switch to newest model releases automatically
- Increase model parameters để memorize more information

**Correct Answer:** Update document collection và monitor performance metrics

**Explanation:** Best practice là update underlying documents (new policies, procedures) và continuously monitor performance. Retraining from scratch expensive, auto-switching risky, và increasing parameters doesn't solve information freshness.

---

## Question 20: integration_architecture
**Title:** System Architecture Integration

**Description:** Kiểm tra hiểu biết về integrating Hugging Face models vào larger systems.

**Question:** Khi integrate Hugging Face Q&A system vào web application, architecture pattern nào recommended?

**Options:**
- Load model trong mỗi web request để ensure latest version
- Use microservice architecture với dedicated model serving component
- Embed model directly trong database để reduce latency
- Cache all possible questions và answers trong memory

**Correct Answer:** Use microservice architecture với dedicated model serving component

**Explanation:** Microservice architecture separates model serving từ web application, enabling better scalability, maintenance, và resource management. Loading model per request inefficient, database embedding inappropriate, full caching impractical.

---
