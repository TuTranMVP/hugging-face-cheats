"""
Demo: Working with Models from Hugging Face Hub
Minh họa các concepts từ working_models.md
"""

import os

from transformers.pipelines import pipeline  # type: ignore[import-untyped]


def demo_text_classification():
    """Demo 1: Text Classification Pipeline"""
    print('🔍 Demo 1: Text Classification')
    print('-' * 40)

    # Tạo pipeline cho phân loại cảm xúc
    classifier = pipeline(
        'text-classification', model='distilbert-base-uncased-finetuned-sst-2-english'
    )

    # Test với các câu tiếng Anh
    texts = [
        'DataCamp is awesome!',
        'I hate bugs in my code',
        'Python is a great programming language',
    ]

    for text in texts:
        result = classifier(text)
        print(f"Text: '{text}'")
        print(f'Result: {result[0]["label"]} (confidence: {result[0]["score"]:.3f})')
        print()


def demo_text_generation():
    """Demo 2: Text Generation with Parameters"""
    print('🚀 Demo 2: Text Generation')
    print('-' * 40)

    # Tạo pipeline cho text generation
    generator = pipeline('text-generation', model='gpt2')

    prompt = 'Artificial intelligence will'

    # Generate với tham số tùy chỉnh
    results = generator(
        prompt,
        max_length=30,  # Giới hạn 30 tokens
        num_return_sequences=2,  # Tạo 2 variations
        do_sample=True,  # Cho phép random sampling
        temperature=0.7,  # Kiểm soát creativity
    )

    print(f"Prompt: '{prompt}'")
    print('Generated texts:')
    for i, result in enumerate(results, 1):
        print(f'{i}. {result["generated_text"]}')
    print()


def demo_save_and_load_model():
    """Demo 3: Save and Load Model Locally"""
    print('💾 Demo 3: Save and Load Model')
    print('-' * 40)

    model_dir = './saved_sentiment_model'

    # Tạo pipeline
    classifier = pipeline(
        'text-classification', model='distilbert-base-uncased-finetuned-sst-2-english'
    )

    # Save model locally
    print('Saving model locally...')
    classifier.save_pretrained(model_dir)
    print(f'✅ Model saved to: {model_dir}')

    # Load model từ local
    print('Loading model from local directory...')
    local_classifier = pipeline('text-classification', model=model_dir)

    # Test model đã load
    test_text = 'Local model loading works perfectly!'
    result = local_classifier(test_text)
    print(
        f'Test with local model: {result[0]["label"]} (score: {result[0]["score"]:.3f})'
    )

    # Kiểm tra kích thước folder
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(model_dir)
        for filename in filenames
    )
    print(f'📁 Model size: {total_size / (1024 * 1024):.1f} MB')
    print()


def demo_multilingual_pipeline():
    """Demo 4: Multilingual Model"""
    print('🌍 Demo 4: Multilingual Capabilities')
    print('-' * 40)

    # Sử dụng model đa ngôn ngữ
    classifier = pipeline(
        'text-classification', model='nlptown/bert-base-multilingual-uncased-sentiment'
    )

    # Test với nhiều ngôn ngữ
    multilingual_texts = [
        ('English', 'This product is amazing!'),
        ('Vietnamese', 'Sản phẩm này rất tuyệt vời!'),
        ('French', 'Ce produit est incroyable!'),
        ('Spanish', '¡Este producto es increíble!'),
    ]

    for lang, text in multilingual_texts:
        try:
            result = classifier(text)
            sentiment = (
                'Positive' if result[0]['label'] in ['POSITIVE', 'POS'] else 'Negative'
            )
            print(f"{lang}: '{text}' → {sentiment} ({result[0]['score']:.3f})")
        except Exception as e:
            print(f'{lang}: Error - {e}')
    print()


if __name__ == '__main__':
    print('🤗 Hugging Face Models Demo')
    print('=' * 50)

    try:
        # Chạy các demo
        demo_text_classification()
        demo_text_generation()
        demo_save_and_load_model()
        demo_multilingual_pipeline()

        print('✅ All demos completed successfully!')
        print('\n📚 Tham khảo thêm tại working_models.md')

    except Exception as e:
        print(f'❌ Error: {e}')
        print('💡 Tip: Đảm bảo đã cài đặt: pip install transformers torch')
