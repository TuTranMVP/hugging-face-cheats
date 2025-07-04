#!/usr/bin/env python3
"""
AI Interview Agent - Hugging Face Knowledge Assessment
CLI Chatbox để phân tích dữ liệu từ file .md và mô phỏng phỏng vấn
"""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
import click
import colorama
import markdown
import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Khởi tạo colorama cho Windows
colorama.init(autoreset=True)

# Khởi tạo Rich console
console = Console()


@dataclass
class Question:
    """Cấu trúc dữ liệu cho câu hỏi"""

    id: str
    title: str
    description: str
    question: str
    options: List[str]
    correct_answer: str
    explanation: str
    difficulty: str = 'medium'
    topic: str = 'general'


@dataclass
class Knowledge:
    """Cấu trúc dữ liệu cho kiến thức"""

    title: str
    content: str
    source_file: str
    sections: List[str]
    keywords: List[str]
    folder: str = ''
    file_type: str = 'markdown'
    importance_score: float = 1.0


@dataclass
class WorkspaceConfig:
    """Cấu hình workspace loading"""

    include_folders: Optional[List[str]] = None  # Folders to include
    exclude_folders: Optional[List[str]] = None  # Folders to exclude
    file_patterns: Optional[List[str]] = None  # File patterns to match
    max_file_size: int = 1024 * 1024  # Max file size in bytes
    enable_code_analysis: bool = True  # Analyze .py files
    enable_auto_summary: bool = True  # Auto generate summaries


class WorkspaceLoader:
    """Nâng cấp để nạp toàn bộ workspace thông minh"""

    def __init__(self, config: Optional[WorkspaceConfig] = None):
        self.config = config or WorkspaceConfig()
        self.md_parser = markdown.Markdown(extensions=['extra', 'toc'])
        self.supported_extensions = {'.md', '.txt', '.py', '.json', '.yml', '.yaml'}

    def discover_workspace(self, root_path: str) -> Dict[str, List[str]]:
        """Khám phá cấu trúc workspace"""
        workspace_structure = {
            'folders': [],
            'markdown_files': [],
            'python_files': [],
            'config_files': [],
            'question_files': [],
            'knowledge_files': [],
        }

        root = Path(root_path)

        for path in root.rglob('*'):
            if path.is_dir():
                if self._should_include_folder(str(path.relative_to(root))):
                    workspace_structure['folders'].append(str(path))
            elif path.is_file():
                rel_path = str(path.relative_to(root))
                if self._should_include_file(rel_path):
                    if path.suffix == '.md':
                        if 'question' in path.name.lower():
                            workspace_structure['question_files'].append(str(path))
                        else:
                            workspace_structure['knowledge_files'].append(str(path))
                        workspace_structure['markdown_files'].append(str(path))
                    elif path.suffix == '.py':
                        workspace_structure['python_files'].append(str(path))
                    elif path.suffix in {'.json', '.yml', '.yaml'}:
                        workspace_structure['config_files'].append(str(path))

        return workspace_structure

    def load_workspace_content(
        self, root_path: str, selected_folders: Optional[List[str]] = None
    ) -> List[Knowledge]:
        """Nạp nội dung workspace với tùy chọn folder"""
        knowledge_base = []

        if selected_folders:
            # Load specific folders
            for folder in selected_folders:
                folder_path = Path(root_path) / folder
                if folder_path.exists():
                    knowledge_base.extend(
                        self._load_folder_content(folder_path, folder)
                    )
        else:
            # Load all workspace
            workspace_structure = self.discover_workspace(root_path)

            # Load markdown files
            for md_file in workspace_structure['knowledge_files']:
                knowledge = self._load_markdown_file(md_file)
                if knowledge:
                    knowledge_base.append(knowledge)

            # Load Python files if enabled
            if self.config.enable_code_analysis:
                for py_file in workspace_structure['python_files']:
                    knowledge = self._load_python_file(py_file)
                    if knowledge:
                        knowledge_base.append(knowledge)

        return knowledge_base

    def _load_folder_content(
        self, folder_path: Path, folder_name: str
    ) -> List[Knowledge]:
        """Nạp nội dung từ một folder cụ thể"""
        knowledge_list = []

        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                if file_path.suffix == '.md':
                    knowledge = self._load_markdown_file(str(file_path))
                elif file_path.suffix == '.py':
                    knowledge = self._load_python_file(str(file_path))
                else:
                    knowledge = self._load_text_file(str(file_path))

                if knowledge:
                    knowledge.folder = folder_name
                    knowledge_list.append(knowledge)

        return knowledge_list

    def _load_markdown_file(self, file_path: str) -> Optional[Knowledge]:
        """Nâng cấp load markdown với phân tích sâu hơn"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Convert markdown to HTML for better parsing
            html_content = self.md_parser.convert(content)
            _soup = BeautifulSoup(html_content, 'html.parser')

            # Extract title
            title = self._extract_title(content, file_path)

            # Extract sections
            sections = self._extract_sections(content)

            # Extract keywords
            keywords = self._extract_keywords(content)

            # Calculate importance score
            importance_score = self._calculate_importance_score(content, file_path)

            return Knowledge(
                title=title,
                content=content,
                source_file=file_path,
                sections=sections,
                keywords=keywords,
                folder=str(Path(file_path).parent.name),
                file_type='markdown',
                importance_score=importance_score,
            )

        except Exception as e:
            console.print(f'[red]Lỗi khi đọc file {file_path}: {e}[/red]')
            return None

    def _load_python_file(self, file_path: str) -> Optional[Knowledge]:
        """Nạp và phân tích file Python"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Extract docstrings and comments
            title = f'Python Code: {Path(file_path).name}'

            # Extract function and class definitions
            sections = self._extract_python_elements(content)

            # Extract technical keywords
            keywords = self._extract_python_keywords(content)

            # Create summarized content
            summary_content = self._create_python_summary(content)

            return Knowledge(
                title=title,
                content=summary_content,
                source_file=file_path,
                sections=sections,
                keywords=keywords,
                folder=str(Path(file_path).parent.name),
                file_type='python',
                importance_score=0.8,  # Slightly lower than markdown
            )

        except Exception as e:
            console.print(f'[red]Lỗi khi đọc file Python {file_path}: {e}[/red]')
            return None

    def _load_text_file(self, file_path: str) -> Optional[Knowledge]:
        """Nạp file text thông thường"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            if len(content) > self.config.max_file_size:
                content = (
                    content[: self.config.max_file_size] + '...\n[Content truncated]'
                )

            title = f'Text File: {Path(file_path).name}'
            keywords = self._extract_keywords(content)

            return Knowledge(
                title=title,
                content=content,
                source_file=file_path,
                sections=[],
                keywords=keywords,
                folder=str(Path(file_path).parent.name),
                file_type='text',
                importance_score=0.5,
            )

        except Exception as e:
            console.print(f'[red]Lỗi khi đọc file text {file_path}: {e}[/red]')
            return None

    def _should_include_folder(self, folder_path: str) -> bool:
        """Kiểm tra có nên include folder không"""
        # Exclude common unnecessary folders
        exclude_defaults = {
            '.git',
            '__pycache__',
            '.vscode',
            'node_modules',
            '.pytest_cache',
        }

        folder_name = Path(folder_path).name
        if folder_name in exclude_defaults:
            return False

        if self.config.exclude_folders:
            if any(excluded in folder_path for excluded in self.config.exclude_folders):
                return False

        if self.config.include_folders:
            return any(
                included in folder_path for included in self.config.include_folders
            )

        return True

    def _should_include_file(self, file_path: str) -> bool:
        """Kiểm tra có nên include file không"""
        path = Path(file_path)

        # Check file size
        try:
            if path.stat().st_size > self.config.max_file_size:
                return False
        except:  # noqa: E722
            return False

        # Check extension
        if path.suffix not in self.supported_extensions:
            return False

        # Check patterns
        if self.config.file_patterns:
            return any(pattern in file_path for pattern in self.config.file_patterns)

        return True

    def _extract_title(self, content: str, file_path: str) -> str:
        """Trích xuất title thông minh"""
        # Try to find first H1 heading
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()

        # Try to find title in front matter
        frontmatter_match = re.search(r'^---\s*\ntitle:\s*(.+)$', content, re.MULTILINE)
        if frontmatter_match:
            return frontmatter_match.group(1).strip()

        # Use filename as fallback
        return Path(file_path).stem.replace('_', ' ').replace('-', ' ').title()

    def _extract_sections(self, content: str) -> List[str]:
        """Trích xuất các sections từ markdown"""
        sections = []

        # Find all headings
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        for match in re.finditer(heading_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            sections.append(f'{"  " * (level - 1)}{title}')

        return sections

    def _extract_keywords(self, content: str) -> List[str]:
        """Trích xuất keywords thông minh"""
        keywords = set()

        # Technical terms (capitalize words)
        tech_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]*)*\b', content)
        keywords.update(tech_terms[:20])

        # Common ML/AI terms
        ml_terms = [
            'hugging face',
            'transformers',
            'model',
            'pipeline',
            'tokenizer',
            'dataset',
            'training',
            'inference',
            'api',
            'nlp',
            'llm',
            'bert',
            'gpt',
            'embedding',
            'fine-tuning',
            'classification',
            'summarization',
        ]

        content_lower = content.lower()
        for term in ml_terms:
            if term in content_lower:
                keywords.add(term)

        # Extract quoted terms
        quoted_terms = re.findall(r'["\']([^"\']+)["\']', content)
        keywords.update(
            [term for term in quoted_terms if len(term) > 2 and len(term) < 30]
        )

        return list(keywords)[:30]  # Limit to 30 keywords

    def _calculate_importance_score(self, content: str, file_path: str) -> float:
        """Tính điểm quan trọng của file"""
        score = 1.0

        # File name indicators
        filename = Path(file_path).name.lower()
        if 'readme' in filename:
            score += 0.5
        if 'introduction' in filename:
            score += 0.3
        if 'question' in filename:
            score += 0.4

        # Content quality indicators
        if len(content) > 1000:
            score += 0.2
        if '```' in content:  # Has code blocks
            score += 0.1
        if content.count('#') > 3:  # Well structured
            score += 0.1

        return min(score, 2.0)  # Cap at 2.0

    def _extract_python_elements(self, content: str) -> List[str]:
        """Trích xuất các elements từ Python code"""
        elements = []

        # Find functions
        func_pattern = r'def\s+(\w+)\s*\([^)]*\):'
        for match in re.finditer(func_pattern, content):
            elements.append(f'Function: {match.group(1)}')

        # Find classes
        class_pattern = r'class\s+(\w+)(?:\([^)]*\))?:'
        for match in re.finditer(class_pattern, content):
            elements.append(f'Class: {match.group(1)}')

        return elements

    def _extract_python_keywords(self, content: str) -> List[str]:
        """Trích xuất keywords từ Python code"""
        keywords = set()

        # Import statements
        import_pattern = r'(?:from\s+(\S+)\s+)?import\s+([^#\n]+)'
        for match in re.finditer(import_pattern, content):
            if match.group(1):
                keywords.add(match.group(1))
            imports = match.group(2).split(',')
            keywords.update([imp.strip() for imp in imports])

        # Function and class names
        func_class_pattern = r'(?:def|class)\s+(\w+)'
        for match in re.finditer(func_class_pattern, content):
            keywords.add(match.group(1))

        return list(keywords)[:20]

    def _create_python_summary(self, content: str) -> str:
        """Tạo summary cho Python file"""
        lines = content.split('\n')
        summary_lines = []

        # Add docstring if exists
        if '"""' in content:
            in_docstring = False
            for line in lines:
                if '"""' in line and not in_docstring:
                    in_docstring = True
                    summary_lines.append(line)
                elif '"""' in line and in_docstring:
                    summary_lines.append(line)
                    break
                elif in_docstring:
                    summary_lines.append(line)

        # Add function/class definitions
        for line in lines:
            if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                summary_lines.append(line)

        return '\n'.join(summary_lines[:50])  # Limit to 50 lines


class MarkdownParser:
    """Parser để phân tích file markdown"""

    def __init__(self):
        self.md = markdown.Markdown(extensions=['extra', 'toc'])

    def parse_questions(self, file_path: str) -> List[Question]:
        """Phân tích file markdown chứa câu hỏi"""
        questions = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Tách các câu hỏi bằng dấu ---
            question_blocks = content.split('---')

            for block in question_blocks:
                if 'exercise_' in block and (
                    '**Question:**' in block or '**Câu hỏi:**' in block
                ):
                    question = self._parse_question_block(block)
                    if question:
                        questions.append(question)

        except Exception as e:
            console.print(f'[red]Lỗi khi đọc file {file_path}: {e}[/red]')

        return questions

    def _parse_question_block(self, block: str) -> Optional[Question]:
        """Phân tích một block câu hỏi"""
        try:
            # Tìm exercise ID
            exercise_match = re.search(r'exercise_(\w+)', block)
            if not exercise_match:
                return None

            exercise_id = exercise_match.group(1)

            # Tìm title (hỗ trợ cả tiếng Anh và tiếng Việt)
            title_match = re.search(r'\*\*(?:Title|Tiêu đề):\*\*\s*(.+)', block)
            title = (
                title_match.group(1).strip()
                if title_match
                else f'Question {exercise_id}'
            )

            # Tìm description (hỗ trợ cả tiếng Anh và tiếng Việt)
            desc_match = re.search(r'\*\*(?:Description|Mô tả):\*\*\s*(.+)', block)
            description = desc_match.group(1).strip() if desc_match else ''

            # Tìm question (hỗ trợ cả tiếng Anh và tiếng Việt)
            question_match = re.search(r'\*\*(?:Question|Câu hỏi):\*\*\s*(.+)', block)
            question = question_match.group(1).strip() if question_match else ''

            # Tìm options (hỗ trợ cả tiếng Anh và tiếng Việt)
            options_section = re.search(
                r'\*\*(?:Options|Các lựa chọn):\*\*\s*\n((?:- .+\n?)+)', block
            )
            options = []
            if options_section:
                option_lines = options_section.group(1).strip().split('\n')
                options = [
                    line.strip('- ').strip()
                    for line in option_lines
                    if line.strip().startswith('- ')
                ]

            # Tìm correct answer (hỗ trợ cả tiếng Anh và tiếng Việt)
            answer_match = re.search(
                r'\*\*(?:Correct Answer|Đáp án đúng):\*\*\s*(.+)', block
            )
            correct_answer = answer_match.group(1).strip() if answer_match else ''

            # Tìm explanation (hỗ trợ cả tiếng Anh và tiếng Việt)
            explanation_match = re.search(
                r'\*\*(?:Explanation|Giải thích):\*\*\s*(.+)', block
            )
            explanation = (
                explanation_match.group(1).strip() if explanation_match else ''
            )

            return Question(
                id=exercise_id,
                title=title,
                description=description,
                question=question,
                options=options,
                correct_answer=correct_answer,
                explanation=explanation,
            )

        except Exception as e:
            console.print(f'[red]Lỗi khi phân tích câu hỏi: {e}[/red]')
            return None

    def parse_knowledge(self, file_path: str) -> Knowledge:
        """Phân tích file markdown chứa kiến thức"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse markdown
            html = self.md.convert(content)
            soup = BeautifulSoup(html, 'html.parser')

            # Lấy title (h1 đầu tiên)
            title_tag = soup.find('h1')
            title = title_tag.get_text().strip() if title_tag else Path(file_path).stem

            # Lấy các section (h2, h3)
            sections = []
            for heading in soup.find_all(['h2', 'h3']):
                sections.append(heading.get_text().strip())

            # Tạo keywords từ headings và content
            keywords = []
            text_content = soup.get_text()

            # Thêm từ khóa từ headings
            keywords.extend([s.lower() for s in sections])

            # Tìm từ khóa quan trọng (có thể cải thiện bằng NLP)
            important_words = re.findall(
                r'\b(?:Hugging Face|LLM|Model|API|Hub|Dataset|Fine-tuning|Transformer|GPT|BERT|Llama)\b',
                text_content,
                re.IGNORECASE,
            )
            keywords.extend([w.lower() for w in important_words])

            return Knowledge(
                title=title,
                content=content,
                source_file=file_path,
                sections=sections,
                keywords=list(set(keywords)),  # Remove duplicates
            )

        except Exception as e:
            console.print(f'[red]Lỗi khi đọc file {file_path}: {e}[/red]')
            return None  # type: ignore


@dataclass
class AIResponse:
    """Cấu trúc phản hồi từ AI"""

    content: str
    source: str = 'ai'
    confidence: float = 0.0
    thinking_process: str = ''
    knowledge_used: Optional[List[str]] = None

    def __post_init__(self):
        if self.knowledge_used is None:
            self.knowledge_used = []


class OllamaAI:
    """Tích hợp AI Ollama llama3:8b"""

    def __init__(
        self, model_name: str = 'llama3:8b', base_url: str = 'http://localhost:11434'
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.is_available = False
        self.session = requests.Session()
        self._check_availability()

    def _check_availability(self):
        """Kiểm tra Ollama có khả dụng không"""
        try:
            response = self.session.get(f'{self.base_url}/api/tags', timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                if self.model_name in available_models:
                    self.is_available = True
                    console.print(f'[green]✓ Ollama {self.model_name} khả dụng[/green]')
                else:
                    console.print(
                        f'[yellow]⚠ Model {self.model_name} chưa được cài đặt[/yellow]'
                    )
                    console.print(
                        f'[dim]Có thể cài đặt: ollama pull {self.model_name}[/dim]'
                    )
            else:
                console.print('[yellow]⚠ Ollama server không phản hồi[/yellow]')
        except requests.RequestException:
            console.print(
                '[yellow]⚠ Không thể kết nối tới Ollama (http://localhost:11434)[/yellow]'
            )
            console.print('[dim]Hãy chắc chắn Ollama đã chạy: ollama serve[/dim]')

    def generate_response(
        self, prompt: str, context: str = '', max_tokens: int = 500
    ) -> AIResponse:
        """Tạo phản hồi từ AI"""
        if not self.is_available:
            return AIResponse(
                content='❌ AI Ollama không khả dụng. Sử dụng chế độ rule-based.',
                source='fallback',
                confidence=0.0,
            )

        try:
            # Tạo prompt template
            system_prompt = self._create_system_prompt(context)
            full_prompt = f'{system_prompt}\n\nQuestion: {prompt}\n\nAnswer:'

            # Gọi API Ollama
            payload = {
                'model': self.model_name,
                'prompt': full_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'top_k': 40,
                    'top_p': 0.9,
                    'num_predict': max_tokens,
                },
            }

            response = self.session.post(
                f'{self.base_url}/api/generate', json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '').strip()

                # Parse response để extract thinking process
                thinking_process = ''
                knowledge_used = []

                if '<thinking>' in ai_response:
                    thinking_match = re.search(
                        r'<thinking>(.*?)</thinking>', ai_response, re.DOTALL
                    )
                    if thinking_match:
                        thinking_process = thinking_match.group(1).strip()
                        ai_response = re.sub(
                            r'<thinking>.*?</thinking>',
                            '',
                            ai_response,
                            flags=re.DOTALL,
                        ).strip()

                # Tính confidence dựa trên độ dài và chất lượng response
                confidence = min(0.9, len(ai_response) / 300)

                return AIResponse(
                    content=ai_response,
                    source='ollama',
                    confidence=confidence,
                    thinking_process=thinking_process,
                    knowledge_used=knowledge_used,
                )
            else:
                console.print(f'[red]Lỗi API Ollama: {response.status_code}[/red]')
                return AIResponse(
                    content='❌ Lỗi khi gọi API Ollama', source='error', confidence=0.0
                )

        except requests.RequestException as e:
            console.print(f'[red]Lỗi kết nối Ollama: {e}[/red]')
            return AIResponse(
                content='❌ Không thể kết nối tới Ollama',
                source='error',
                confidence=0.0,
            )

    def _create_system_prompt(self, context: str) -> str:
        """Tạo system prompt cho AI"""
        return f"""Bạn là AI Assistant chuyên về Hugging Face và Machine Learning.

Context từ Knowledge Base:
{context}

QUAN TRỌNG - Quy tắc trả lời:
1. Trả lời bằng tiếng Việt, ngắn gọn và chính xác
2. KHÔNG sử dụng markdown formatting (**, *, `, #, etc.)
3. Sử dụng emoji phù hợp để làm cho câu trả lời sinh động
4. Cung cấp thông tin thực tế dựa trên context
5. Nếu cần suy luận, bao quanh trong <thinking></thinking>
6. Trả lời trực tiếp, tránh từ ngữ thừa

Format trả lời mong muốn:
- Câu trả lời trực tiếp với emoji
- Danh sách dùng dấu • thay vì số
- Ví dụ cụ thể khi có thể
- Tối đa 3-4 ý chính

Ví dụ tốt:
🤖 Hugging Face là nền tảng AI mở với các tính năng:
• Model Hub: Lưu trữ hàng ngàn models
• Datasets: Bộ sưu tập dữ liệu training
• Spaces: Demo ứng dụng AI
• Transformers: Thư viện Python dễ sử dụng

Ví dụ tránh:
**Hugging Face** là một *platform* quan trọng...

Hãy trả lời ngắn gọn, dễ hiểu và thực tế."""


class InterviewAgent:
    """AI Agent để mô phỏng phỏng vấn - Enhanced với workspace loading"""

    def __init__(self):
        self.questions = []
        self.knowledge_base = []
        self.current_question = None
        self.score = 0
        self.answered_questions = []
        self.session_stats = {
            'total_questions': 0,
            'correct_answers': 0,
            'start_time': None,
            'end_time': None,
        }
        self.workspace_loader = WorkspaceLoader()
        self.workspace_structure = {}

    def load_workspace(
        self, root_path: str, config: Optional[WorkspaceConfig] = None
    ) -> Dict[str, Any]:
        """Nạp toàn bộ workspace với configuration"""
        if config:
            self.workspace_loader.config = config

        console.print(f'\n[bold blue]🔍 Khám phá workspace: {root_path}[/bold blue]')

        # Discover workspace structure
        self.workspace_structure = self.workspace_loader.discover_workspace(root_path)

        # Display workspace structure
        self._display_workspace_structure()

        # Ask user which folders to load
        selected_folders = self._select_folders_to_load()

        # Load content
        with Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task('Đang nạp workspace...', total=100)

            # Load knowledge base
            progress.update(task, description='Nạp knowledge base...', completed=20)
            self.knowledge_base = self.workspace_loader.load_workspace_content(
                root_path, selected_folders
            )

            # Load questions from discovered question files
            progress.update(task, description='Nạp câu hỏi...', completed=60)
            parser = MarkdownParser()
            for question_file in self.workspace_structure['question_files']:
                questions = parser.parse_questions(question_file)
                self.questions.extend(questions)

            progress.update(task, description='Hoàn thành!', completed=100)

        # Display loading results
        self._display_loading_results()

        return {
            'knowledge_count': len(self.knowledge_base),
            'question_count': len(self.questions),
            'folders_loaded': selected_folders or 'all',
            'workspace_structure': self.workspace_structure,
        }

    def _display_workspace_structure(self):
        """Hiển thị cấu trúc workspace"""
        table = Table(title='📁 Cấu trúc Workspace', title_style='bold blue')
        table.add_column('Loại', style='cyan', width=20)
        table.add_column('Số lượng', justify='right', style='magenta')
        table.add_column('Chi tiết', style='white')

        table.add_row(
            '📂 Folders',
            str(len(self.workspace_structure['folders'])),
            f'Có thể chọn: {len(self.workspace_structure["folders"])}',
        )
        table.add_row(
            '📝 Markdown Files',
            str(len(self.workspace_structure['markdown_files'])),
            'Tài liệu và hướng dẫn',
        )
        table.add_row(
            '❓ Question Files',
            str(len(self.workspace_structure['question_files'])),
            'Ngân hàng câu hỏi',
        )
        table.add_row(
            '🐍 Python Files',
            str(len(self.workspace_structure['python_files'])),
            'Source code và examples',
        )
        table.add_row(
            '⚙️ Config Files',
            str(len(self.workspace_structure['config_files'])),
            'Configuration files',
        )

        console.print(table)

    def _select_folders_to_load(self) -> Optional[List[str]]:
        """Cho phép user chọn folders để load"""
        if not self.workspace_structure['folders']:
            return None

        # Extract unique folder names from full paths
        folder_names = set()
        for folder_path in self.workspace_structure['folders']:
            # Get relative folder names
            parts = Path(folder_path).parts
            if len(parts) > 1:  # Not root
                folder_names.add(parts[-1])  # Last part (folder name)

        folder_list = sorted(folder_names)

        if not folder_list:
            return None

        console.print(
            f'\n[bold yellow]📂 Có {len(folder_list)} folders có thể chọn:[/bold yellow]'
        )

        # Display options
        table = Table(show_header=True, header_style='bold magenta')
        table.add_column('Số', style='cyan', width=5)
        table.add_column('Folder', style='white')
        table.add_column('Mô tả', style='dim')

        folder_descriptions = {
            'getting-started': 'Hướng dẫn cơ bản',
            'pipelines': 'Các pipeline chuyên biệt',
            'text-classification': 'Phân loại văn bản',
            'text-summarization': 'Tóm tắt văn bản',
            'datasets': 'Xử lý dữ liệu',
            '__pycache__': 'Cache files (không khuyến khích)',
        }

        for i, folder in enumerate(folder_list, 1):
            description = folder_descriptions.get(folder, 'Nội dung khác')
            table.add_row(str(i), folder, description)

        table.add_row('0', '[ALL]', 'Tất cả folders')

        console.print(table)

        # Get user choice
        try:
            choice = Prompt.ask(
                '\n[bold]Chọn folders (ví dụ: 1,2,3 hoặc 0 cho tất cả)[/bold]',
                default='0',
            )

            if choice == '0':
                console.print('[green]✓ Sẽ nạp tất cả folders[/green]')
                return None  # Load all

            # Parse selected folders
            selected_indices = [
                int(x.strip()) for x in choice.split(',') if x.strip().isdigit()
            ]
            selected_folders = [
                folder_list[i - 1]
                for i in selected_indices
                if 1 <= i <= len(folder_list)
            ]

            if selected_folders:
                console.print(f'[green]✓ Sẽ nạp: {", ".join(selected_folders)}[/green]')
                return selected_folders
            else:
                console.print(
                    '[yellow]⚠ Không có lựa chọn hợp lệ, sẽ nạp tất cả[/yellow]'
                )
                return None

        except (ValueError, KeyboardInterrupt):
            console.print('[yellow]⚠ Lựa chọn không hợp lệ, sẽ nạp tất cả[/yellow]')
            return None

    def _display_loading_results(self):
        """Hiển thị kết quả loading"""
        console.print('\n[bold green]✅ Đã nạp workspace thành công![/bold green]')

        # Knowledge statistics
        if self.knowledge_base:
            kb_stats = self._analyze_knowledge_base()

            stats_table = Table(
                title='📊 Thống kê Knowledge Base', title_style='bold green'
            )
            stats_table.add_column('Thông tin', style='cyan')
            stats_table.add_column('Giá trị', justify='right', style='white')

            stats_table.add_row('📚 Tổng tài liệu', str(len(self.knowledge_base)))
            stats_table.add_row('📝 Markdown files', str(kb_stats['markdown_count']))
            stats_table.add_row('🐍 Python files', str(kb_stats['python_count']))
            stats_table.add_row('📁 Folders', str(len(kb_stats['folders'])))
            stats_table.add_row('🔑 Keywords', str(kb_stats['total_keywords']))
            stats_table.add_row(
                '📄 Tổng nội dung', f'{kb_stats["total_content_length"]:,} ký tự'
            )

            console.print(stats_table)

            # Top folders by content
            if kb_stats['folder_stats']:
                folder_table = Table(
                    title='📂 Top Folders theo nội dung', title_style='bold blue'
                )
                folder_table.add_column('Folder', style='cyan')
                folder_table.add_column('Files', justify='right', style='magenta')
                folder_table.add_column('Content Size', justify='right', style='white')

                for folder, stats in sorted(
                    kb_stats['folder_stats'].items(),
                    key=lambda x: x[1]['content_length'],
                    reverse=True,
                )[:5]:
                    folder_table.add_row(
                        folder,
                        str(stats['file_count']),
                        f'{stats["content_length"]:,} chars',
                    )

                console.print(folder_table)

        # Question statistics
        if self.questions:
            console.print(
                f'[bold yellow]❓ Đã nạp {len(self.questions)} câu hỏi từ {len(self.workspace_structure["question_files"])} files[/bold yellow]'
            )

    def _analyze_knowledge_base(self) -> Dict[str, Any]:
        """Phân tích knowledge base để có thống kê"""
        stats = {
            'markdown_count': 0,
            'python_count': 0,
            'folders': set(),
            'total_keywords': 0,
            'total_content_length': 0,
            'folder_stats': {},
        }

        for knowledge in self.knowledge_base:
            # Count by file type
            if knowledge.file_type == 'markdown':
                stats['markdown_count'] += 1
            elif knowledge.file_type == 'python':
                stats['python_count'] += 1

            # Collect folders
            stats['folders'].add(knowledge.folder)

            # Count keywords
            stats['total_keywords'] += len(knowledge.keywords)

            # Count content length
            stats['total_content_length'] += len(knowledge.content)

            # Folder statistics
            folder = knowledge.folder
            if folder not in stats['folder_stats']:
                stats['folder_stats'][folder] = {'file_count': 0, 'content_length': 0}

            stats['folder_stats'][folder]['file_count'] += 1
            stats['folder_stats'][folder]['content_length'] += len(knowledge.content)

        return stats

    def load_data(self, file_paths: List[str]):
        """Tải dữ liệu từ các file markdown (method cũ, giữ để backward compatibility)"""
        parser = MarkdownParser()

        with Progress(
            SpinnerColumn(),
            TextColumn('[progress.description]{task.description}'),
            console=console,
        ) as progress:
            task = progress.add_task('Đang tải dữ liệu...', total=len(file_paths))

            for file_path in file_paths:
                if not os.path.exists(file_path):
                    console.print(f'[red]File không tồn tại: {file_path}[/red]')
                    continue

                progress.update(task, description=f'Đang xử lý {Path(file_path).name}')

                if 'question' in file_path.lower():
                    # File chứa câu hỏi
                    questions = parser.parse_questions(file_path)
                    self.questions.extend(questions)
                    console.print(
                        f'[green]✓ Đã tải {len(questions)} câu hỏi từ {Path(file_path).name}[/green]'
                    )
                else:
                    # File chứa kiến thức
                    knowledge = parser.parse_knowledge(file_path)
                    if knowledge:
                        self.knowledge_base.append(knowledge)
                        console.print(
                            f'[green]✓ Đã tải kiến thức từ {Path(file_path).name}[/green]'
                        )

                progress.advance(task)

        console.print(
            f'\n[bold green]Tổng cộng: {len(self.questions)} câu hỏi và {len(self.knowledge_base)} tài liệu kiến thức[/bold green]'
        )

    def start_interview(self):
        """Bắt đầu phỏng vấn"""
        if not self.questions:
            console.print('[red]Không có câu hỏi nào để phỏng vấn![/red]')
            return

        self.session_stats['start_time'] = datetime.now()
        self.session_stats['total_questions'] = len(self.questions)

        console.print(
            Panel.fit(
                '[bold blue]🎯 Bắt đầu phỏng vấn Hugging Face![/bold blue]\n'
                f'Tổng số câu hỏi: {len(self.questions)}\n'
                "Gõ 'help' để xem trợ giúp, 'quit' để thoát",
                title='AI Interview Agent',
                border_style='blue',
            )
        )

        # Shuffle questions for randomness

        random.shuffle(self.questions)

        for i, question in enumerate(self.questions, 1):
            console.print(f'\n[bold cyan]Câu hỏi {i}/{len(self.questions)}[/bold cyan]')

            if self.ask_question(question):
                self.score += 1
                self.session_stats['correct_answers'] += 1

        self.session_stats['end_time'] = datetime.now()
        self.show_final_results()

    def ask_question(self, question: Question) -> bool:
        """Đặt câu hỏi và nhận phản hồi"""
        self.current_question = question

        # Hiển thị câu hỏi
        console.print(
            Panel(
                f'[bold]{question.title}[/bold]\n\n'
                f'[dim]{question.description}[/dim]\n\n'
                f'[yellow]{question.question}[/yellow]',
                title=f'Question ID: {question.id}',
                border_style='yellow',
            )
        )

        # Hiển thị các lựa chọn
        table = Table(show_header=True, header_style='bold magenta')
        table.add_column('Lựa chọn', style='cyan', width=10)
        table.add_column('Nội dung', style='white')

        for i, option in enumerate(question.options, 1):
            table.add_row(f'({i})', option)

        console.print(table)

        # Nhận phản hồi từ user
        while True:
            try:
                response = Prompt.ask(
                    "\n[bold]Chọn đáp án của bạn (1-4) hoặc gõ 'help'/'quit'[/bold]",
                    choices=['1', '2', '3', '4', 'help', 'quit', 'h', 'q'],
                    show_choices=False,
                )

                if response.lower() in ['quit', 'q']:
                    console.print('[red]Thoát phỏng vấn...[/red]')
                    sys.exit(0)

                if response.lower() in ['help', 'h']:
                    self.show_help()
                    continue

                # Chuyển đổi lựa chọn thành text
                choice_index = int(response) - 1
                if 0 <= choice_index < len(question.options):
                    user_answer = question.options[choice_index]
                    return self.check_answer(user_answer, question)
                else:
                    console.print('[red]Lựa chọn không hợp lệ![/red]')
                    continue

            except (ValueError, KeyboardInterrupt):
                console.print('[red]Lựa chọn không hợp lệ![/red]')
                continue

    def check_answer(self, user_answer: str, question: Question) -> bool:
        """Kiểm tra đáp án và đưa ra phản hồi"""
        is_correct = user_answer.strip() == question.correct_answer.strip()

        if is_correct:
            console.print(
                Panel(
                    f'[bold green]🎉 Chính xác![/bold green]\n\n'
                    f'[green]Giải thích:[/green] {question.explanation}',
                    title='Kết quả',
                    border_style='green',
                )
            )
        else:
            console.print(
                Panel(
                    f'[bold red]❌ Sai rồi![/bold red]\n\n'
                    f'[red]Đáp án đúng:[/red] {question.correct_answer}\n'
                    f'[red]Đáp án của bạn:[/red] {user_answer}\n\n'
                    f'[yellow]Giải thích:[/yellow] {question.explanation}',
                    title='Kết quả',
                    border_style='red',
                )
            )

        # Hiển thị kiến thức liên quan
        self.show_related_knowledge(question)

        # Hỏi có muốn tiếp tục không
        if not Confirm.ask('\n[bold]Tiếp tục câu hỏi tiếp theo?[/bold]', default=True):
            console.print('[yellow]Kết thúc phỏng vấn...[/yellow]')
            self.show_final_results()
            sys.exit(0)

        return is_correct

    def show_related_knowledge(self, question: Question):
        """Hiển thị kiến thức liên quan đến câu hỏi"""
        # Tìm kiến thức liên quan dựa trên từ khóa
        related_knowledge = []
        question_keywords = (
            question.title.lower().split() + question.question.lower().split()
        )

        for knowledge in self.knowledge_base:
            for keyword in question_keywords:
                if (
                    keyword in knowledge.keywords
                    or keyword in knowledge.content.lower()
                ):
                    related_knowledge.append(knowledge)
                    break

        if related_knowledge:
            console.print('\n[bold blue]📚 Kiến thức liên quan:[/bold blue]')
            for i, knowledge in enumerate(
                related_knowledge[:2], 1
            ):  # Chỉ hiển thị 2 tài liệu đầu
                console.print(
                    f'[dim]{i}. {knowledge.title} (từ {Path(knowledge.source_file).name})[/dim]'
                )

    def show_help(self):
        """Hiển thị trợ giúp"""
        help_text = """
        [bold blue]🔧 Hướng dẫn sử dụng:[/bold blue]

        [yellow]Các lệnh có sẵn:[/yellow]
        • 1-4: Chọn đáp án
        • help (h): Hiển thị trợ giúp này
        • quit (q): Thoát chương trình

        [yellow]Cách thức hoạt động:[/yellow]
        • Chọn số tương ứng với đáp án bạn cho là đúng
        • Hệ thống sẽ kiểm tra và đưa ra giải thích
        • Điểm số và thống kê sẽ được hiển thị cuối phiên

        [yellow]Mẹo:[/yellow]
        • Đọc kỹ mô tả câu hỏi trước khi chọn
        • Chú ý đến từ khóa quan trọng
        • Sử dụng kiến thức từ tài liệu đã tải
        """
        console.print(Panel(help_text, title='Trợ giúp', border_style='blue'))

    def show_final_results(self):
        """Hiển thị kết quả cuối cùng"""
        if self.session_stats['start_time'] and self.session_stats['end_time']:
            duration = self.session_stats['end_time'] - self.session_stats['start_time']
            duration_str = f'{duration.total_seconds():.1f} giây'
        else:
            duration_str = 'N/A'

        total = self.session_stats['total_questions']
        correct = self.session_stats['correct_answers']
        percentage = (correct / total * 100) if total > 0 else 0

        # Đánh giá kết quả
        if percentage >= 80:
            grade = 'Xuất sắc! 🌟'
            color = 'green'
        elif percentage >= 60:
            grade = 'Khá tốt! 👍'
            color = 'yellow'
        elif percentage >= 40:
            grade = 'Cần cải thiện 📚'
            color = 'orange'
        else:
            grade = 'Cần học thêm 💪'
            color = 'red'

        result_table = Table(title='🎯 Kết quả phỏng vấn', title_style='bold blue')
        result_table.add_column('Chỉ số', style='cyan', width=20)
        result_table.add_column('Giá trị', style='white')

        result_table.add_row('Tổng câu hỏi', str(total))
        result_table.add_row('Câu trả lời đúng', str(correct))
        result_table.add_row('Tỷ lệ chính xác', f'{percentage:.1f}%')
        result_table.add_row('Thời gian', duration_str)
        result_table.add_row('Đánh giá', f'[{color}]{grade}[/{color}]')

        console.print('\n')
        console.print(result_table)

        # Lời khuyên
        console.print(
            Panel(
                self.get_advice(percentage), title='💡 Lời khuyên', border_style='blue'
            )
        )

    def get_advice(self, percentage: float) -> str:
        """Đưa ra lời khuyên dựa trên kết quả"""
        if percentage >= 80:
            return '[green]Bạn đã có kiến thức rất tốt về Hugging Face! Tiếp tục duy trì và khám phá các tính năng nâng cao.[/green]'
        elif percentage >= 60:
            return '[yellow]Bạn đã nắm được những kiến thức cơ bản. Hãy tập trung vào những phần còn thiếu và thực hành thêm.[/yellow]'
        elif percentage >= 40:
            return '[orange]Bạn cần đọc lại tài liệu và thực hành thêm. Tập trung vào các khái niệm cơ bản trước.[/orange]'
        else:
            return '[red]Hãy bắt đầu từ những kiến thức cơ bản về Hugging Face. Đọc documentation và làm theo tutorial.[/red]'


class ChatMode:
    """Chế độ chat tương tác với AI Ollama"""

    def __init__(self, agent: InterviewAgent):
        self.agent = agent
        self.conversation_history = []
        self.ollama_ai = OllamaAI()
        self.use_ai = self.ollama_ai.is_available

    def start_chat(self):
        """Bắt đầu chế độ chat"""
        ai_status = '🤖 AI Ollama' if self.use_ai else '📚 Rule-based'
        console.print(
            Panel.fit(
                f'[bold green]💬 Chế độ Chat tương tác ({ai_status})[/bold green]\n'
                'Hỏi tôi bất cứ điều gì về Hugging Face!\n'
                "Gõ 'quit' để thoát, 'interview' để chuyển sang chế độ phỏng vấn\n"
                f"Gõ 'ai' để {'tắt' if self.use_ai else 'bật'} AI mode",
                title='AI Chat Assistant',
                border_style='green',
            )
        )

        while True:
            try:
                user_input = Prompt.ask('\n[bold blue]Bạn[/bold blue]')

                if user_input.lower() in ['quit', 'q', 'exit']:
                    console.print('[green]Tạm biệt! 👋[/green]')
                    break

                if user_input.lower() in ['interview', 'test']:
                    console.print('[yellow]Chuyển sang chế độ phỏng vấn...[/yellow]')
                    self.agent.start_interview()
                    continue

                if user_input.lower() == 'ai':
                    self.use_ai = not self.use_ai
                    status = '🤖 AI Ollama' if self.use_ai else '📚 Rule-based'
                    console.print(f'[yellow]Chuyển sang chế độ: {status}[/yellow]')
                    continue

                # Xử lý câu hỏi
                if self.use_ai:
                    response = self.process_question_with_ai(user_input)
                else:
                    response = self.process_question_rule_based(user_input)

                console.print('\n[bold green]🤖 AI Assistant[/bold green]')
                console.print(Panel(response, border_style='green'))

            except KeyboardInterrupt:
                console.print('\n[green]Tạm biệt! 👋[/green]')
                break

    def _build_context(self, question: str) -> str:
        """Xây dựng context từ knowledge base cho AI"""
        question_lower = question.lower()
        relevant_knowledge = []

        # Tìm kiến thức liên quan
        for knowledge in self.agent.knowledge_base:
            relevance_score = 0

            # Kiểm tra keywords
            for keyword in knowledge.keywords:
                if keyword in question_lower:
                    relevance_score += 2

            # Kiểm tra content
            content_lower = knowledge.content.lower()
            for word in question_lower.split():
                if word in content_lower:
                    relevance_score += 1

            if relevance_score > 0:
                relevant_knowledge.append((relevance_score, knowledge))

        # Sắp xếp theo relevance score
        relevant_knowledge.sort(key=lambda x: x[0], reverse=True)

        # Tạo context
        context = ''
        for _score, knowledge in relevant_knowledge[:3]:  # Lấy top 3
            context += f'## {knowledge.title}\n'
            context += f'Source: {Path(knowledge.source_file).name}\n'
            context += f'Content: {knowledge.content[:800]}...\n\n'

        return (
            context if context else 'Không có thông tin liên quan trong knowledge base.'
        )

    def process_question_with_ai(self, question: str) -> str:
        """Xử lý câu hỏi với AI Ollama"""
        try:
            # Tạo context từ knowledge base
            context = self._build_context(question)

            # Gọi AI
            with console.status('[bold green]🤖 AI đang suy nghĩ...', spinner='dots'):
                ai_response = self.ollama_ai.generate_response(question, context)

            # Format response với template đẹp
            return self._format_ai_response(ai_response, question)

        except Exception as e:
            console.print(f'[red]Lỗi AI: {e}[/red]')
            return self.process_question_rule_based(question)

    def _format_ai_response(self, ai_response: AIResponse, question: str) -> str:
        """Format AI response với template đẹp và clean"""
        lines = []

        # Header với confidence
        confidence_emoji = self._get_confidence_emoji(ai_response.confidence)
        lines.append(
            f'🤖 AI Analysis {confidence_emoji} ({ai_response.confidence:.0%} confidence)'
        )
        lines.append('')

        # Main content - làm sạch và format
        clean_content = self._clean_ai_content(ai_response.content)
        if clean_content:
            lines.append('📋 Answer:')
            lines.append('─' * 50)
            lines.extend(self._format_content_lines(clean_content))
            lines.append('')

        # Thinking process (nếu có) - compact format
        if ai_response.thinking_process:
            thinking_clean = self._clean_ai_content(ai_response.thinking_process)
            if thinking_clean and len(thinking_clean) > 20:  # Chỉ hiện nếu có nội dung
                lines.append('💭 AI Reasoning:')
                lines.append('─' * 30)
                # Rút gọn thinking process
                thinking_summary = self._summarize_thinking(thinking_clean)
                lines.append(f'   {thinking_summary}')
                lines.append('')

        # Footer với source info
        if ai_response.source == 'ollama':
            lines.append('🔍 Analysis based on:')
            lines.append(
                f'   • Knowledge Base: {len(self.agent.knowledge_base)} documents'
            )
            lines.append('   • AI Model: Ollama Llama3')

        return '\n'.join(lines)

    def _get_confidence_emoji(self, confidence: float) -> str:
        """Lấy emoji phù hợp với confidence level"""
        if confidence >= 0.8:
            return '🎯'  # High confidence
        elif confidence >= 0.6:
            return '✅'  # Good confidence
        elif confidence >= 0.4:
            return '⚠️'  # Medium confidence
        else:
            return '❓'  # Low confidence

    def _clean_ai_content(self, content: str) -> str:
        """Làm sạch content AI, loại bỏ markdown và formatting không cần thiết"""
        if not content:
            return ''

        # Loại bỏ markdown formatting
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # **bold** -> bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)  # *italic* -> italic
        content = re.sub(r'`(.*?)`', r'\1', content)  # `code` -> code
        content = re.sub(r'#{1,6}\s*', '', content)  # headings
        content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)  # links

        # Làm sạch các ký tự đặc biệt - CẨN THẬN KHÔNG LẶP
        # Chỉ chuẩn hóa các dấu bullet ở đầu dòng
        content = re.sub(r'^[\s]*[-\*]\s+', '• ', content, flags=re.MULTILINE)
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Loại bỏ line breaks thừa

        # Loại bỏ các thẻ thinking
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)

        return content.strip()

    def _format_content_lines(self, content: str) -> List[str]:
        """Format content thành các dòng đẹp với indentation"""
        lines = content.split('\n')
        formatted = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Bullet points
            if line.startswith('•'):
                formatted.append(f'   {line}')
            # Numbered lists
            elif re.match(r'^\d+\.', line):
                formatted.append(f'   {line}')
            # Headers or important lines
            elif line.isupper() or line.endswith(':'):
                formatted.append(f'\n   {line}')
            # Regular content
            else:
                # Wrap long lines
                if len(line) > 80:
                    wrapped = self._wrap_text(line, 80)
                    for wrapped_line in wrapped:
                        formatted.append(f'   {wrapped_line}')
                else:
                    formatted.append(f'   {line}')

        return formatted

    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text để không quá dài"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def _summarize_thinking(self, thinking: str) -> str:
        """Tóm tắt thinking process để hiển thị gọn"""
        # Lấy câu đầu tiên hoặc 100 ký tự đầu
        sentences = thinking.split('.')
        if sentences and len(sentences[0]) > 20:
            summary = sentences[0].strip()
            if len(summary) > 100:
                summary = summary[:100] + '...'
            return summary
        else:
            return thinking[:100] + '...' if len(thinking) > 100 else thinking

    def process_question_rule_based(self, question: str) -> str:
        """Xử lý câu hỏi theo rule-based (phương pháp cũ)"""
        return self.process_question(question)

    def process_question(self, question: str) -> str:
        """Xử lý câu hỏi và trả lời dựa trên knowledge base"""
        question_lower = question.lower()

        # Tìm kiến thức liên quan
        relevant_knowledge = []
        for knowledge in self.agent.knowledge_base:
            for keyword in knowledge.keywords:
                if keyword in question_lower:
                    relevant_knowledge.append(knowledge)
                    break

        # Nếu không tìm thấy kiến thức liên quan, tìm theo content
        if not relevant_knowledge:
            for knowledge in self.agent.knowledge_base:
                if any(
                    word in knowledge.content.lower() for word in question_lower.split()
                ):
                    relevant_knowledge.append(knowledge)

        if relevant_knowledge:
            return self._format_rule_based_response(relevant_knowledge, question)
        else:
            return self._format_no_results_response()

    def _format_rule_based_response(
        self, relevant_knowledge: List[Knowledge], question: str
    ) -> str:
        """Format rule-based response với template đẹp"""
        lines = []

        # Header
        lines.append('📚 Knowledge Base Search ✅')
        lines.append('')

        # Main content
        lines.append('📋 Found Information:')
        lines.append('─' * 50)

        for i, knowledge in enumerate(
            relevant_knowledge[:2], 1
        ):  # Chỉ lấy 2 tài liệu đầu
            lines.append(f'\n� Source {i}: {self._clean_markdown(knowledge.title)}')
            lines.append('─' * 30)

            # Sử dụng smart search để tìm nội dung liên quan
            smart_results = self._smart_search(question, knowledge)

            if smart_results:
                # Hiển thị kết quả tìm kiếm thông minh
                for result in smart_results[:3]:  # Giới hạn 3 kết quả
                    cleaned_result = self._clean_markdown(result)
                    if cleaned_result and len(cleaned_result) > 10:
                        # Format với indentation - loại bỏ bullet có sẵn
                        formatted_result = self._format_single_result(cleaned_result)
                        # Loại bỏ bullet ở đầu nếu có
                        if formatted_result.startswith('• '):
                            formatted_result = formatted_result[2:]
                        lines.append(f'   • {formatted_result}')
            else:
                # Fallback: extract key points
                key_points = self._extract_key_points(knowledge.content)
                if key_points:
                    for point in key_points[:3]:  # Giới hạn 3 điểm
                        lines.append(f'   • {point}')
                else:
                    # Final fallback: clean content summary
                    summary = self._create_summary(knowledge.content)
                    lines.append(f'   {summary}')

            lines.append(f'\n   📁 From: {Path(knowledge.source_file).name}')

        # Footer
        lines.append('')
        lines.append('🔍 Search based on:')
        lines.append('   • Keyword matching and content analysis')
        lines.append(f'   • Knowledge Base: {len(self.agent.knowledge_base)} documents')

        return '\n'.join(lines)

    def _format_no_results_response(self) -> str:
        """Format response khi không tìm thấy kết quả"""
        lines = []

        lines.append('📚 Knowledge Base Search ❌')
        lines.append('')
        lines.append('📋 No Direct Match Found')
        lines.append('─' * 50)
        lines.append('')
        lines.append('💡 Suggestions:')
        lines.append('   • Try asking about: Hugging Face, Models, Hub, API')
        lines.append("   • Switch to interview mode: type 'interview'")
        lines.append("   • Toggle AI mode: type 'ai'")
        lines.append('')
        lines.append('🔍 Available Resources:')
        lines.append(f'   • Knowledge Base: {len(self.agent.knowledge_base)} documents')
        lines.append(f'   • Question Bank: {len(self.agent.questions)} questions')

        return '\n'.join(lines)

    def _format_single_result(self, result: str) -> str:
        """Format một kết quả tìm kiếm"""
        # Loại bỏ ký tự thừa và format đẹp
        result = result.strip()

        # Nếu quá dài, cắt ngắn
        if len(result) > 120:
            result = result[:120] + '...'

        return result

    def _create_summary(self, content: str) -> str:
        """Tạo summary ngắn gọn từ content"""
        # Lấy câu đầu tiên hoặc đoạn đầu
        sentences = content.split('.')
        if sentences and len(sentences[0].strip()) > 20:
            summary = sentences[0].strip()
            if len(summary) > 150:
                summary = summary[:150] + '...'
            return summary
        else:
            # Fallback: lấy đoạn đầu
            paragraphs = content.split('\n\n')
            if paragraphs:
                first_para = paragraphs[0].strip()
                if len(first_para) > 200:
                    first_para = first_para[:200] + '...'
                return self._clean_markdown(first_para)

        return 'Content available but requires specific keywords to search.'

    def _clean_markdown(self, text: str) -> str:
        """Loại bỏ các ký tự markdown formatting"""
        if not text:
            return ''

        # Loại bỏ markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold** -> bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # *italic* -> italic
        text = re.sub(r'`(.*?)`', r'\1', text)  # `code` -> code
        text = re.sub(r'#{1,6}\s*', '', text)  # ## heading -> heading
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # [text](url) -> text
        text = re.sub(
            r'^\s*[-*+]\s*', '• ', text, flags=re.MULTILINE
        )  # - item -> • item

        return text.strip()

    def _format_content(self, content: str) -> str:
        """Format nội dung để hiển thị đẹp hơn"""
        lines = content.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if line:
                if line.startswith('•'):
                    formatted_lines.append(line)
                else:
                    formatted_lines.append(f'{line}')

        return '\n'.join(formatted_lines[:4])  # Giới hạn 4 dòng

    def _smart_search(self, question: str, knowledge: Knowledge) -> List[str]:
        """Tìm kiếm thông minh trong knowledge base"""
        question_words = set(question.lower().split())
        lines = knowledge.content.split('\n')
        scored_lines = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Tính điểm relevance
            line_words = set(line.lower().split())
            score = len(question_words.intersection(line_words))

            # Bonus điểm cho dòng chứa từ khóa quan trọng
            if any(
                keyword in line.lower()
                for keyword in ['hugging face', 'model', 'api', 'hub']
            ):
                score += 2

            if score > 0:
                scored_lines.append((score, line))

        # Sort theo điểm và lấy top results
        scored_lines.sort(key=lambda x: x[0], reverse=True)
        return [line for score, line in scored_lines[:5]]

    def _extract_key_points(self, content: str) -> List[str]:
        """Trích xuất các điểm chính từ nội dung"""
        lines = content.split('\n')
        key_points = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Tìm các bullet points
            if line.startswith(('- ', '• ', '* ')):
                cleaned = self._clean_markdown(line[2:].strip())
                if cleaned:
                    key_points.append(cleaned)

            # Tìm các heading quan trọng
            elif line.startswith('###'):
                cleaned = self._clean_markdown(line[3:].strip())
                if cleaned:
                    key_points.append(f'📌 {cleaned}')

        return key_points[:6]  # Giới hạn 6 điểm chính


@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
@click.option(
    '--mode',
    '-m',
    type=click.Choice(['interview', 'chat', 'both']),
    default='both',
    help='Chọn chế độ: interview (phỏng vấn), chat (trò chuyện), both (cả hai)',
)
@click.option('--shuffle', '-s', is_flag=True, help='Xáo trộn thứ tự câu hỏi')
@click.option('--limit', '-l', type=int, help='Giới hạn số lượng câu hỏi')
@click.option('--verbose', '-v', is_flag=True, help='Hiển thị thông tin chi tiết')
@click.option(
    '--workspace',
    '-w',
    type=click.Path(exists=True),
    help='Đường dẫn workspace để load toàn bộ',
)
@click.option(
    '--folders',
    '-f',
    type=str,
    help='Danh sách folders cách nhau bởi dấu phẩy (vd: getting-started,pipelines)',
)
@click.option('--include-python', is_flag=True, help='Bao gồm phân tích file Python')
@click.option(
    '--max-file-size',
    type=int,
    default=1024 * 1024,
    help='Kích thước file tối đa (bytes)',
)
@click.option(
    '--exclude-folders',
    type=str,
    help='Loại trừ folders (vd: __pycache__,node_modules)',
)
def main(
    files,
    mode,
    shuffle,
    limit,
    verbose,
    workspace,
    folders,
    include_python,
    max_file_size,
    exclude_folders,
):
    """
    AI Interview Agent - Hugging Face Knowledge Assessment

    Phân tích các file .md và tạo phiên phỏng vấn tương tác.

    Examples:
        # Traditional file-based loading
        python main.py getting-started/questions.md getting-started/introduction.md
        python main.py *.md --mode chat

        # New workspace loading
        python main.py --workspace . --mode chat
        python main.py --workspace . --folders 'getting-started,pipelines'
        python main.py --workspace . --include-python --exclude-folders '__pycache__'
    """

    # Banner
    console.print(
        Panel.fit(
            '[bold blue]🤖 AI Interview Agent[/bold blue]\n'
            '[dim]Hugging Face Knowledge Assessment[/dim]\n'
            f'[green]Mode: {mode.upper()}[/green]',
            title='Welcome',
            border_style='blue',
        )
    )

    # Khởi tạo agent
    agent = InterviewAgent()

    # Configure workspace loading if enabled
    if workspace or folders or include_python:
        # Configure workspace loader
        config = WorkspaceConfig()
        config.enable_code_analysis = include_python
        config.max_file_size = max_file_size

        if exclude_folders:
            config.exclude_folders = exclude_folders.split(',')

        agent.workspace_loader.config = config

        # Determine workspace path
        workspace_path = workspace or '.'

        # Parse folders if specified
        selected_folders = folders.split(',') if folders else None

        # Load workspace content
        console.print(f'\n[bold]🌐 Loading workspace: {workspace_path}[/bold]')
        if selected_folders:
            console.print(
                f'[yellow]📁 Selected folders: {", ".join(selected_folders)}[/yellow]'
            )

        knowledge_base = agent.workspace_loader.load_workspace_content(
            workspace_path, selected_folders or []
        )

        # Set knowledge base and load questions
        agent.knowledge_base = knowledge_base
        agent.questions = []

        # Try to load questions from workspace structure
        workspace_structure = agent.workspace_loader.discover_workspace(workspace_path)
        parser = MarkdownParser()

        for question_file in workspace_structure['question_files']:
            questions = parser.parse_questions(question_file)
            agent.questions.extend(questions)

        console.print(
            f'[green]✓ Loaded {len(knowledge_base)} documents and {len(agent.questions)} questions[/green]'
        )

    elif files:
        # Traditional file-based loading
        console.print(f'\n[bold]📁 Loading {len(files)} file(s)...[/bold]')
        agent.load_data(list(files))

        if verbose:
            console.print(
                f'\n[dim]Loaded {len(agent.questions)} questions and {len(agent.knowledge_base)} documents[/dim]'
            )

    else:
        # Auto-discovery mode
        console.print('\n[yellow]🔍 Auto-discovery mode[/yellow]')
        console.print('[dim]Looking for markdown files in current directory...[/dim]')

        workspace_structure = agent.workspace_loader.discover_workspace('.')
        markdown_files = workspace_structure['markdown_files']

        if markdown_files:
            console.print(f'[green]Found {len(markdown_files)} markdown files[/green]')

            # Load the discovered files
            agent.load_data(markdown_files[:10])  # Limit to first 10 files
            console.print(f'[green]✓ Loaded {len(agent.questions)} questions[/green]')
        else:
            console.print('[red]❌ No markdown files found![/red]')
            console.print('\n[yellow]Examples:[/yellow]')
            console.print('  python main.py getting-started/questions.md')
            console.print('  python main.py --workspace . --mode chat')
            return

    # Apply limit
    if limit and limit < len(agent.questions):
        agent.questions = agent.questions[:limit]
        console.print(f'[yellow]Question limit applied: {limit}[/yellow]')

    # Apply shuffle
    if shuffle and agent.questions:
        import random

        random.shuffle(agent.questions)
        console.print('[yellow]Questions shuffled[/yellow]')

    # Chế độ hoạt động
    if mode == 'interview':
        agent.start_interview()
    elif mode == 'chat':
        chat = ChatMode(agent)
        chat.start_chat()
    else:  # both
        console.print('\n[bold yellow]Chọn chế độ hoạt động:[/bold yellow]')
        console.print('1. 🎯 Phỏng vấn (Interview)')
        console.print('2. 💬 Trò chuyện (Chat)')
        console.print('3. 🔄 Cả hai (Both)')

        choice = Prompt.ask('Chọn chế độ', choices=['1', '2', '3'], default='1')

        if choice == '1':
            agent.start_interview()
        elif choice == '2':
            chat = ChatMode(agent)
            chat.start_chat()
        else:
            chat = ChatMode(agent)
            console.print(
                "\n[bold]Bắt đầu với chế độ chat, gõ 'interview' để chuyển sang phỏng vấn[/bold]"
            )
            chat.start_chat()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print('\n[green]Tạm biệt! 👋[/green]')
    except Exception as e:
        console.print(f'\n[red]Lỗi: {e}[/red]')
        if '--verbose' in sys.argv:
            import traceback

            console.print(traceback.format_exc())
