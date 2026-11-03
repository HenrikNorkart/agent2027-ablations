from typing import Literal

import pandas as pd
from rich.console import Console, Style
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskID
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
import rich.box


from rogueone.utils import console_cfg
from rogueone.utils.singelton import Singleton
import re


Color = Literal["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]
Language = Literal[
    "python",
    "java",
    "javascript",
    "html",
    "css",
    "json",
    "yaml",
    "bash",
    "sql",
]


def disabler():
    def inner(func):
        def wrapper(*args, **kwargs):
            return (
                func(*args, **kwargs)
                if not console_cfg.disable
                else args[1] if len(args) > 1 else None
            )

        return wrapper

    return inner


def tool_feedback(message: str | None) -> str | None:
    """Value a tool returns to the agent for an error/rejection site.

    ROGUE_ONE_TOOL_ERROR_FEEDBACK=message returns the text; =none returns None,
    reproducing the t17 behaviour. Independent of CONSOLE_DISABLE.
    """
    from rogueone.utils import rouge_one_cfg

    return None if rouge_one_cfg.tool_error_feedback == "none" else message


def truncate_tool_output(text: str, max_chars: int | None = None) -> str:
    """Cut a tool result to ROGUE_ONE_TOOL_OUTPUT_MAX_CHARS (0 = unbounded)."""
    if max_chars is None:
        from rogueone.utils import rouge_one_cfg

        max_chars = rouge_one_cfg.tool_output_max_chars
    if not max_chars or max_chars <= 0 or len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n[... output truncated: {len(text) - max_chars} of {len(text)} "
        f"characters omitted (limit {max_chars}) ...]"
    )


class ConsoleManager(metaclass=Singleton):
    def __init__(self):
        self._console = Console()
        self._task_id = None
        self._progress_bar: Progress = None

    @disabler()
    @staticmethod
    def reset_progress_bar():
        ConsoleManager()._progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )

    @staticmethod
    def console_print(message: str, style: Style | str | None = None) -> str:
        # Always returns the message (E2): the return value no longer depends on
        # CONSOLE_DISABLE. Tool sites route it through tool_feedback().
        if not console_cfg.disable:
            ConsoleManager()._console.log(message, style=style)
        return message

    @disabler()
    @staticmethod
    def _get_code_panel(
        code: str, language: Language = "python", title: str = ""
    ) -> Panel:
        if "python" in language:
            code = re.sub(r";\s*", "\n", code)
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        return Panel(syntax, title=title, border_style="green")

    @disabler()
    @staticmethod
    def console_rule(title: str = "", color: Color = "cyan") -> None:
        ConsoleManager()._console.rule(title, style=Style(color=color))

    @staticmethod
    def console_error_print(
        message: str, code: str | None = None, language: Language = "python"
    ) -> str:
        # Always returns the message (E2); see console_print.
        if console_cfg.disable:
            return message
        try:
            ConsoleManager()._console.log(
                f"[bold red]Error:[/bold red] {message}",
                (
                    ConsoleManager._get_code_panel(code, language)
                    if code is not None
                    else ""
                ),
            )
        except Exception as e:
            ConsoleManager()._console.log(f"[bold red]Error:[/bold red] {e}")
        return message

    @disabler()
    @staticmethod
    def console_agent_logging(
        agent_name: str,
        tool_name: str,
        message: str = "",
        code: str | None = None,
        language: Language = "python",
        post_message: str = "",
    ) -> None:
        try:
            ConsoleManager()._console.log(
                f"[bold blue]{agent_name}.{tool_name}()[/bold blue] \n{message}",
                ConsoleManager._get_code_panel(code, language) if code else "",
                f"[bold green]{post_message}[/bold green]",
            )
        except Exception as e:
            ConsoleManager()._console.log(f"[bold red]Error:[/bold red] {e}")

    @disabler()
    @staticmethod
    def progress_bar_add_task(description: str, total: int | None = None) -> TaskID:
        ConsoleManager.reset_progress_bar()
        ConsoleManager()._task_id = ConsoleManager()._progress_bar.add_task(
            description=description, total=total
        )
        return ConsoleManager()._task_id

    @disabler()
    @staticmethod
    def progress_bar_update(
        task_id: int | TaskID | None = None,
        advance: float | None = None,
        completed: float | None = None,
        total: int | None = None,
        description: str | None = None,
    ) -> None:
        instance = ConsoleManager()
        assert (
            task_id is not None or instance._task_id is not None
        ), "Task ID must be provided to update the progress bar."

        instance._progress_bar.update(
            task_id or instance._task_id,
            advance=advance,
            completed=completed,
            total=total,
            description=description,
        )

    @disabler()
    @staticmethod
    def print_dataframe_as_table(
        df: pd.DataFrame,
        title: str = "",
        style: Style | str | None = None,
    ) -> None:
        ConsoleManager.print_table(
            title=title,
            columns=list(df.columns),
            rows=df.values.tolist(),
            style=style,
        )

    @disabler()
    @staticmethod
    def print_dict_as_table(
        data: dict,
        title: str = "",
        columns=("Key", "Value"),
        style: Style | str | None = None,
    ) -> None:
        rows = []
        for k, v in data.items():
            if isinstance(v, (list, dict, set, tuple)):
                rows.append([k, *[str(i) for i in v]])
            else:
                rows.append([k, v])

        ConsoleManager.print_table(
            title=title,
            columns=list(columns),
            rows=rows,
            style=style,
        )

    @disabler()
    @staticmethod
    def print_table(
        title: str,
        columns: list[str],
        rows: list[list[str]],
        style: Style | str | None = None,
    ) -> None:
        table = Table(title=title, box=rich.box.SQUARE, style=style)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*map(str, row), end_section=True)
        ConsoleManager()._console.print(table)

    @property
    def console(self) -> Console:
        return ConsoleManager()._console

    @property
    def progress_bar(self) -> Progress:
        return ConsoleManager()._progress_bar   