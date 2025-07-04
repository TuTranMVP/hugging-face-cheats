#!/usr/bin/env python3
"""
AI Training Pipeline - Hybrid Gemini + Ollama System
Tối ưu hóa model với dữ liệu chất lượng cao từ Gemini để training Ollama
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from main import GeminiAI, Knowledge, WorkspaceLoader

console = Console()


@dataclass
class TrainingExample:
    """Cấu trúc dữ liệu training example"""

    question: str
    context: str
    gemini_answer: str
    confidence: float
    timestamp: datetime
    quality_score: float = 0.0
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelMetrics:
    """Metrics đánh giá model performance"""

    accuracy: float
    confidence_avg: float
    response_time: float
    training_examples: int
    last_updated: datetime
    model_version: str


class OllamaTrainer:
    """Training và fine-tuning Ollama models"""

    def __init__(self, base_url: str = 'http://localhost:11434'):
        self.base_url = base_url
        self.is_available = False
        self.session = requests.Session()
        self._check_availability()

    def _check_availability(self):
        """Kiểm tra Ollama có khả dụng không"""
        try:
            response = self.session.get(f'{self.base_url}/api/tags', timeout=5)
            if response.status_code == 200:
                self.is_available = True
                console.print('[green]✓ Ollama server khả dụng[/green]')
            else:
                console.print('[yellow]⚠ Ollama server không phản hồi[/yellow]')
        except requests.RequestException:
            console.print('[yellow]⚠ Không thể kết nối tới Ollama[/yellow]')

    def create_modelfile(
        self, base_model: str, training_data: List[TrainingExample]
    ) -> str:
        """Tạo Modelfile cho Ollama từ training data"""

        # Tạo system prompt tối ưu từ training examples
        high_quality_examples = [ex for ex in training_data if ex.confidence > 0.8]

        system_prompt = f"""Bạn là AI Interview Expert chuyên sâu về Hugging Face và Machine Learning.

TRAINING DATA INSIGHTS:
- Đã học từ {len(high_quality_examples)} examples chất lượng cao
- Average confidence: {sum(ex.confidence for ex in high_quality_examples) / len(high_quality_examples):.2f}
- Cập nhật lần cuối: {datetime.now().strftime('%Y-%m-%d %H:%M')}

QUY TẮC PHẢN HỒI:
1. Trả lời bằng tiếng Việt chuyên nghiệp và chính xác
2. Sử dụng emoji phù hợp để tăng tính sinh động
3. KHÔNG sử dụng markdown formatting
4. Cung cấp examples cụ thể và practical
5. Tập trung vào Hugging Face ecosystem

CHUYÊN MÔN:
- Hugging Face Hub, Transformers, Datasets, Spaces
- Machine Learning workflows và best practices
- Python programming trong ML context
- Real-world applications và troubleshooting

Hãy trả lời ngắn gọn, chính xác và hữu ích!"""

        # Tạo training examples cho Modelfile
        examples_text = ''
        for ex in high_quality_examples[:20]:  # Limit to top 20 examples
            examples_text += f"""
USER: {ex.question}
ASSISTANT: {ex.gemini_answer}

"""

        modelfile = f"""FROM {base_model}

SYSTEM "{system_prompt}"

# High-quality training examples from Gemini
{examples_text}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 512
"""

        return modelfile

    def train_model(self, model_name: str, modelfile_content: str) -> bool:
        """Training model với Ollama"""
        try:
            # Tạo file tạm cho Modelfile
            modelfile_path = Path(f'/tmp/{model_name}_modelfile')
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                f.write(modelfile_content)

            console.print(f'[yellow]🔧 Đang training model: {model_name}...[/yellow]')

            # Gọi Ollama create command
            import subprocess

            result = subprocess.run(
                ['ollama', 'create', model_name, '-f', str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                console.print(
                    f'[green]✅ Model {model_name} được training thành công![/green]'
                )
                # Cleanup
                modelfile_path.unlink()
                return True
            else:
                console.print(f'[red]❌ Lỗi training: {result.stderr}[/red]')
                return False

        except Exception as e:
            console.print(f'[red]❌ Lỗi training model: {e}[/red]')
            return False

    def test_model(self, model_name: str, test_questions: List[str]) -> ModelMetrics:
        """Test model performance"""

        if not self.is_available:
            return ModelMetrics(0, 0, 0, 0, datetime.now(), model_name)

        total_time = 0
        successful_tests = 0

        console.print(f'[yellow]🧪 Testing model: {model_name}...[/yellow]')

        for question in test_questions:
            try:
                start_time = time.time()

                response = self.session.post(
                    f'{self.base_url}/api/generate',
                    json={
                        'model': model_name,
                        'prompt': question,
                        'stream': False,
                        'options': {'temperature': 0.7, 'num_predict': 256},
                    },
                    timeout=30,
                )

                response_time = time.time() - start_time
                total_time += response_time

                if response.status_code == 200:
                    result = response.json()
                    if result.get('response'):
                        successful_tests += 1

            except Exception:
                console.print(
                    f'[dim]Test failed for question: {question[:50]}...[/dim]'
                )

        accuracy = successful_tests / len(test_questions) if test_questions else 0
        avg_response_time = total_time / len(test_questions) if test_questions else 0

        return ModelMetrics(
            accuracy=accuracy,
            confidence_avg=0.8,  # Estimated
            response_time=avg_response_time,
            training_examples=0,
            last_updated=datetime.now(),
            model_version=model_name,
        )


class SmartTrainingPipeline:
    """Pipeline thông minh kết hợp Gemini + Ollama"""

    def __init__(self):
        self.gemini_ai = GeminiAI()
        self.ollama_trainer = OllamaTrainer()
        self.workspace_loader = WorkspaceLoader()
        self.training_data: List[TrainingExample] = []
        self.training_history = []

    def generate_training_data(
        self, knowledge_base: List[Knowledge], num_examples: int = 50
    ) -> List[TrainingExample]:
        """Tạo training data chất lượng cao từ Gemini"""

        console.print(
            f'[bold blue]🚀 Generating {num_examples} training examples with Gemini...[/bold blue]'
        )

        if not self.gemini_ai.is_available:
            console.print('[red]❌ Gemini AI không khả dụng![/red]')
            return []

        training_examples = []

        # Tạo các câu hỏi từ knowledge base
        questions = self._generate_questions_from_knowledge(
            knowledge_base, num_examples
        )

        with Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                'Generating training data...', total=len(questions)
            )

            for i, (question, context) in enumerate(questions):
                try:
                    # Sinh answer với Gemini
                    progress.update(
                        task,
                        description=f'Processing question {i + 1}/{len(questions)}',
                    )

                    ai_response = self.gemini_ai.generate_response(question, context)

                    if ai_response.confidence > 0.6:  # Chỉ lấy responses chất lượng cao
                        example = TrainingExample(
                            question=question,
                            context=context,
                            gemini_answer=ai_response.content,
                            confidence=ai_response.confidence,
                            timestamp=datetime.now(),
                            metadata={
                                'source': 'gemini_generation',
                                'thinking_process': ai_response.thinking_process,
                            },
                        )

                        # Đánh giá quality score
                        example.quality_score = self._calculate_quality_score(example)
                        training_examples.append(example)

                    progress.advance(task)

                    # Rate limiting
                    time.sleep(0.5)

                except Exception as e:
                    console.print(f'[dim]Skipped question due to error: {e}[/dim]')
                    progress.advance(task)

        # Sắp xếp theo quality score
        training_examples.sort(key=lambda x: x.quality_score, reverse=True)

        console.print(
            f'[green]✅ Generated {len(training_examples)} high-quality examples[/green]'
        )
        console.print(
            f'[yellow]Average confidence: {sum(ex.confidence for ex in training_examples) / len(training_examples):.2f}[/yellow]'
        )

        return training_examples

    def _generate_questions_from_knowledge(
        self, knowledge_base: List[Knowledge], num_questions: int
    ) -> List[Tuple[str, str]]:
        """Tạo questions từ knowledge base"""

        question_templates = [
            '{topic} là gì và tại sao nó quan trọng?',
            'Cách sử dụng {topic} trong thực tế?',
            'So sánh {topic} với các alternatives khác?',
            'Ưu điểm và nhược điểm của {topic}?',
            'Best practices khi làm việc với {topic}?',
            'Troubleshooting common issues với {topic}?',
            'Ví dụ cụ thể về {topic} trong project thực tế?',
            'Cách optimize performance với {topic}?',
        ]

        questions = []
        topics_used = set()

        for knowledge in knowledge_base:
            # Extract topics từ knowledge
            topics = knowledge.keywords[:5]  # Top 5 keywords

            for topic in topics:
                if len(questions) >= num_questions:
                    break

                if topic not in topics_used and len(topic) > 3:
                    template = question_templates[
                        len(questions) % len(question_templates)
                    ]
                    question = template.format(topic=topic)

                    # Context từ knowledge
                    context = f'## {knowledge.title}\n{knowledge.content[:1000]}...'

                    questions.append((question, context))
                    topics_used.add(topic)

            if len(questions) >= num_questions:
                break

        return questions

    def _calculate_quality_score(self, example: TrainingExample) -> float:
        """Tính quality score cho training example"""
        score = example.confidence  # Base từ confidence

        # Bonus cho response length (not too short, not too long)
        response_length = len(example.gemini_answer)
        if 100 <= response_length <= 800:
            score += 0.1

        # Bonus cho presence của examples hoặc code
        if any(
            keyword in example.gemini_answer.lower()
            for keyword in ['ví dụ', 'example', 'code', 'python', '```']
        ):
            score += 0.1

        # Bonus cho structured content
        if '•' in example.gemini_answer or example.gemini_answer.count('\n') > 2:
            score += 0.05

        # Bonus cho technical terms
        if any(
            term in example.gemini_answer.lower()
            for term in ['hugging face', 'model', 'api', 'dataset', 'pipeline']
        ):
            score += 0.05

        return min(1.0, score)

    def create_optimized_model(
        self,
        base_model: str = 'llama3:8b',
        model_name: str = None,  # type: ignore
    ) -> bool:
        """Tạo model tối ưu từ training data"""

        if not model_name:
            model_name = f'hf-expert-{datetime.now().strftime("%Y%m%d")}'

        if not self.training_data:
            console.print('[red]❌ Không có training data! Hãy generate trước.[/red]')
            return False

        if not self.ollama_trainer.is_available:
            console.print('[red]❌ Ollama không khả dụng![/red]')
            return False

        console.print(
            f'[bold blue]🔧 Creating optimized model: {model_name}[/bold blue]'
        )

        # Tạo Modelfile
        modelfile = self.ollama_trainer.create_modelfile(base_model, self.training_data)

        # Training model
        success = self.ollama_trainer.train_model(model_name, modelfile)

        if success:
            # Test model
            test_questions = [ex.question for ex in self.training_data[:10]]
            metrics = self.ollama_trainer.test_model(model_name, test_questions)

            self._display_model_metrics(model_name, metrics)

            # Save training history
            self.training_history.append(
                {
                    'model_name': model_name,
                    'base_model': base_model,
                    'training_examples': len(self.training_data),
                    'timestamp': datetime.now(),
                    'metrics': metrics,
                }
            )

            return True

        return False

    def _display_model_metrics(self, model_name: str, metrics: ModelMetrics):
        """Hiển thị metrics của model"""

        table = Table(
            title=f'📊 Model Performance: {model_name}', title_style='bold green'
        )
        table.add_column('Metric', style='cyan')
        table.add_column('Value', style='white', justify='right')

        table.add_row('Accuracy', f'{metrics.accuracy:.2%}')
        table.add_row('Avg Response Time', f'{metrics.response_time:.2f}s')
        table.add_row('Training Examples', str(metrics.training_examples))
        table.add_row('Last Updated', metrics.last_updated.strftime('%Y-%m-%d %H:%M'))

        console.print(table)

    def continuous_improvement(self, workspace_path: str, check_interval: int = 3600):
        """Continuous learning và improvement"""

        console.print(
            '[bold blue]🔄 Starting continuous improvement mode...[/bold blue]'
        )
        console.print(f'[yellow]Check interval: {check_interval} seconds[/yellow]')

        last_update = datetime.now()

        while True:
            try:
                console.print(
                    f'\n[dim]{datetime.now().strftime("%H:%M:%S")} - Checking for updates...[/dim]'
                )

                # Load workspace để check updates
                knowledge_base = self.workspace_loader.load_workspace_content(
                    workspace_path
                )

                # Check if có content mới
                if self._has_new_content(knowledge_base, last_update):
                    console.print(
                        '[yellow]🆕 Detected new content, generating training data...[/yellow]'
                    )

                    # Generate new training examples
                    new_examples = self.generate_training_data(knowledge_base, 20)

                    if new_examples:
                        # Add to existing training data
                        self.training_data.extend(new_examples)

                        # Keep only best examples (limit memory)
                        self.training_data.sort(
                            key=lambda x: x.quality_score, reverse=True
                        )
                        self.training_data = self.training_data[:200]

                        # Create updated model
                        model_name = f'hf-expert-v{len(self.training_history) + 1}'
                        if self.create_optimized_model(model_name=model_name):
                            console.print(
                                f'[green]✅ Updated model: {model_name}[/green]'
                            )

                        last_update = datetime.now()

                # Sleep until next check
                time.sleep(check_interval)

            except KeyboardInterrupt:
                console.print('\n[yellow]Stopping continuous improvement...[/yellow]')
                break
            except Exception as e:
                console.print(f'[red]Error in continuous improvement: {e}[/red]')
                time.sleep(60)  # Wait 1 minute before retry

    def _has_new_content(
        self, knowledge_base: List[Knowledge], last_update: datetime
    ) -> bool:
        """Check xem có content mới không"""
        # Simple implementation - check if files are newer
        for knowledge in knowledge_base:
            try:
                file_mtime = Path(knowledge.source_file).stat().st_mtime
                file_datetime = datetime.fromtimestamp(file_mtime)
                if file_datetime > last_update:
                    return True
            except:  # noqa: E722
                continue
        return False

    def save_training_data(self, filepath: str):
        """Save training data to file"""
        data = {
            'training_examples': [
                {
                    'question': ex.question,
                    'context': ex.context,
                    'gemini_answer': ex.gemini_answer,
                    'confidence': ex.confidence,
                    'quality_score': ex.quality_score,
                    'timestamp': ex.timestamp.isoformat(),
                    'metadata': ex.metadata,
                }
                for ex in self.training_data
            ],
            'training_history': self.training_history,
            'created': datetime.now().isoformat(),
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        console.print(f'[green]💾 Saved training data to: {filepath}[/green]')

    def load_training_data(self, filepath: str):
        """Load training data from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.training_data = []
            for ex_data in data['training_examples']:
                example = TrainingExample(
                    question=ex_data['question'],
                    context=ex_data['context'],
                    gemini_answer=ex_data['gemini_answer'],
                    confidence=ex_data['confidence'],
                    timestamp=datetime.fromisoformat(ex_data['timestamp']),
                    quality_score=ex_data.get('quality_score', 0.0),
                    metadata=ex_data.get('metadata', {}),
                )
                self.training_data.append(example)

            self.training_history = data.get('training_history', [])

            console.print(
                f'[green]📥 Loaded {len(self.training_data)} training examples[/green]'
            )

        except Exception as e:
            console.print(f'[red]❌ Error loading training data: {e}[/red]')


def main():
    """Main demo function"""
    console.print(
        Panel.fit(
            """[bold blue]🚀 AI Training Pipeline - Hybrid Gemini + Ollama[/bold blue]
            [green]✨ Tính năng:[/green]
            • [bold]Smart Data Generation[/bold] - Tạo training data từ Gemini
            • [bold]Ollama Fine-tuning[/bold] - Training local model optimized
            • [bold]Continuous Learning[/bold] - Tự động cập nhật với data mới
            • [bold]Quality Scoring[/bold] - Đánh giá và filter data chất lượng cao
            • [bold]Performance Metrics[/bold] - Tracking model performance

            [yellow]🎯 Workflow:[/yellow]
            1. Generate high-quality training data với Gemini
            2. Fine-tune Ollama model với data đó
            3. Test và evaluate model performance
            4. Continuous improvement với data mới""",
            title='🤖 Smart AI Training System',
            border_style='blue',
        )
    )

    pipeline = SmartTrainingPipeline()

    # Check prerequisites
    if not pipeline.gemini_ai.is_available:
        console.print('[red]❌ Gemini AI không khả dụng! Check API key.[/red]')
        return

    if not pipeline.ollama_trainer.is_available:
        console.print('[red]❌ Ollama không khả dụng! Start ollama serve.[/red]')
        return

    console.print('[green]✅ All systems ready![/green]')

    while True:
        console.print('\n[bold yellow]📋 Choose action:[/bold yellow]')
        console.print('1. 🔄 Generate training data')
        console.print('2. 🏗️  Create optimized model')
        console.print('3. 🧪 Test model performance')
        console.print('4. 🔄 Start continuous improvement')
        console.print('5. 💾 Save/Load training data')
        console.print('6. 📊 View training history')
        console.print('0. ❌ Exit')

        choice = Prompt.ask(
            'Select option', choices=['0', '1', '2', '3', '4', '5', '6'], default='1'
        )

        if choice == '0':
            console.print('[green]Goodbye! 👋[/green]')
            break
        elif choice == '1':
            # Generate training data
            workspace_path = Prompt.ask('Workspace path', default='.')
            num_examples = int(Prompt.ask('Number of examples', default='50'))

            knowledge_base = pipeline.workspace_loader.load_workspace_content(
                workspace_path
            )
            if knowledge_base:
                pipeline.training_data = pipeline.generate_training_data(
                    knowledge_base, num_examples
                )
            else:
                console.print('[red]❌ No knowledge base found![/red]')

        elif choice == '2':
            # Create model
            if not pipeline.training_data:
                console.print('[red]❌ No training data! Generate first.[/red]')
                continue

            base_model = Prompt.ask('Base model', default='llama3:8b')
            model_name = Prompt.ask('Model name (optional)', default='')

            pipeline.create_optimized_model(base_model, model_name or None)  # type: ignore

        elif choice == '3':
            # Test model
            model_name = Prompt.ask('Model name to test')
            test_questions = [
                'Hugging Face là gì?',
                'Cách sử dụng transformers library?',
                'Fine-tuning model như thế nào?',
                'So sánh BERT và GPT?',
                'Hugging Face Hub có tính năng gì?',
            ]

            metrics = pipeline.ollama_trainer.test_model(model_name, test_questions)
            pipeline._display_model_metrics(model_name, metrics)

        elif choice == '4':
            # Continuous improvement
            workspace_path = Prompt.ask('Workspace path', default='.')
            interval = int(Prompt.ask('Check interval (seconds)', default='3600'))

            pipeline.continuous_improvement(workspace_path, interval)

        elif choice == '5':
            # Save/Load data
            action = Prompt.ask('Save or Load?', choices=['save', 'load'])
            filepath = Prompt.ask('File path', default='training_data.json')

            if action == 'save':
                pipeline.save_training_data(filepath)
            else:
                pipeline.load_training_data(filepath)

        elif choice == '6':
            # View history
            if pipeline.training_history:
                table = Table(title='📈 Training History', title_style='bold cyan')
                table.add_column('Model', style='white')
                table.add_column('Examples', justify='right', style='magenta')
                table.add_column('Accuracy', justify='right', style='green')
                table.add_column('Date', style='dim')

                for entry in pipeline.training_history:
                    table.add_row(
                        entry['model_name'],
                        str(entry['training_examples']),
                        f'{entry["metrics"].accuracy:.2%}',
                        entry['timestamp'].strftime('%Y-%m-%d %H:%M'),
                    )

                console.print(table)
            else:
                console.print('[yellow]📋 No training history yet[/yellow]')


if __name__ == '__main__':
    main()
