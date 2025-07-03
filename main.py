#!/usr/bin/env python3
"""
AI Interview Agent - Hugging Face Knowledge Assessment
CLI Chatbox để phân tích dữ liệu từ file .md và mô phỏng phỏng vấn
"""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import sys
from typing import List, Optional

from bs4 import BeautifulSoup
import click
import colorama
import markdown
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
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
                if 'exercise_' in block and '**Question:**' in block:
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

            # Tìm title
            title_match = re.search(r'\*\*Title:\*\*\s*(.+)', block)
            title = (
                title_match.group(1).strip()
                if title_match
                else f'Question {exercise_id}'
            )

            # Tìm description
            desc_match = re.search(r'\*\*Description:\*\*\s*(.+)', block)
            description = desc_match.group(1).strip() if desc_match else ''

            # Tìm question
            question_match = re.search(r'\*\*Question:\*\*\s*(.+)', block)
            question = question_match.group(1).strip() if question_match else ''

            # Tìm options
            options_section = re.search(r'\*\*Options:\*\*\s*\n((?:- .+\n?)+)', block)
            options = []
            if options_section:
                option_lines = options_section.group(1).strip().split('\n')
                options = [
                    line.strip('- ').strip()
                    for line in option_lines
                    if line.strip().startswith('- ')
                ]

            # Tìm correct answer
            answer_match = re.search(r'\*\*Correct Answer:\*\*\s*(.+)', block)
            correct_answer = answer_match.group(1).strip() if answer_match else ''

            # Tìm explanation
            explanation_match = re.search(r'\*\*Explanation:\*\*\s*(.+)', block)
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


class InterviewAgent:
    """AI Agent để mô phỏng phỏng vấn"""

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

    def load_data(self, file_paths: List[str]):
        """Tải dữ liệu từ các file markdown"""
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
        import random

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
    """Chế độ chat tương tác"""

    def __init__(self, agent: InterviewAgent):
        self.agent = agent
        self.conversation_history = []

    def start_chat(self):
        """Bắt đầu chế độ chat"""
        console.print(
            Panel.fit(
                '[bold green]💬 Chế độ Chat tương tác[/bold green]\n'
                'Hỏi tôi bất cứ điều gì về Hugging Face!\n'
                "Gõ 'quit' để thoát, 'interview' để chuyển sang chế độ phỏng vấn",
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

                # Xử lý câu hỏi
                response = self.process_question(user_input)
                console.print('\n[bold green]🤖 AI Assistant[/bold green]')
                console.print(Panel(response, border_style='green'))

            except KeyboardInterrupt:
                console.print('\n[green]Tạm biệt! 👋[/green]')
                break

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
            # Tạo phản hồi dựa trên kiến thức
            response = 'Dựa trên tài liệu đã tải, tôi có thể trả lời:\n\n'

            for knowledge in relevant_knowledge[:2]:  # Chỉ lấy 2 tài liệu đầu
                response += f'**{knowledge.title}**\n'
                # Trích xuất đoạn văn liên quan
                lines = knowledge.content.split('\n')
                relevant_lines = []
                for line in lines:
                    if any(word in line.lower() for word in question_lower.split()):
                        relevant_lines.append(line.strip())

                if relevant_lines:
                    response += '\n'.join(relevant_lines[:3])  # Chỉ lấy 3 dòng đầu
                else:
                    response += knowledge.content[:300] + '...'  # Lấy 300 ký tự đầu

                response += f'\n\n*(Nguồn: {Path(knowledge.source_file).name})*\n\n'
        else:
            response = 'Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu đã tải. Hãy thử:\n\n'
            response += "• Chuyển sang chế độ phỏng vấn: gõ 'interview'\n"
            response += '• Hỏi về các chủ đề như: Hugging Face, LLM, Model, Hub, API\n'
            response += f'• Tôi có {len(self.agent.knowledge_base)} tài liệu và {len(self.agent.questions)} câu hỏi'

        return response


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
def main(files, mode, shuffle, limit, verbose):
    """
    AI Interview Agent - Hugging Face Knowledge Assessment

    Phân tích các file .md và tạo phiên phỏng vấn tương tác.

    Examples:
        python main.py getting-started/questions.md getting-started/introduction.md
        python main.py *.md --mode chat
        python main.py questions.md --mode interview --limit 5
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

    if not files:
        console.print('[red]❌ Cần chỉ định ít nhất một file .md![/red]')
        console.print('\n[yellow]Ví dụ:[/yellow]')
        console.print(
            '  python main.py getting-started/questions.md getting-started/introduction.md'
        )
        console.print('  python main.py *.md --mode chat')
        return

    # Khởi tạo agent
    agent = InterviewAgent()

    # Tải dữ liệu
    console.print(f'\n[bold]📁 Tải dữ liệu từ {len(files)} file(s)...[/bold]')
    agent.load_data(list(files))

    if verbose:
        console.print(
            f'\n[dim]Loaded {len(agent.questions)} questions and {len(agent.knowledge_base)} documents[/dim]'
        )

    # Áp dụng giới hạn
    if limit and limit < len(agent.questions):
        agent.questions = agent.questions[:limit]
        console.print(f'[yellow]Giới hạn số câu hỏi: {limit}[/yellow]')

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
