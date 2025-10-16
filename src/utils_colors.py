class Colors:
    """ANSI color codes for terminal output."""
    
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Text styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    
    # Reset
    RESET = '\033[0m'
    
    # Combined styles for specific purposes
    HEADER = BOLD + MAGENTA
    SYSTEM = BOLD + BLUE
    USER_PROMPT = BOLD + GREEN
    ERROR = BOLD + RED
    SUCCESS = BOLD + GREEN
    INFO = CYAN
    WARNING = YELLOW
    SEPARATOR = MAGENTA


class ColorFormatter:
    """Handles colored output formatting for the restaurant system."""
    
    def __init__(self, use_colors=False):
        """
        Initialize color formatter.
        
        Input: use_colors (bool) - Whether to use colored output
        """
        self.use_colors = use_colors
    
    def format_text(self, text, color_code="", reset_after=True):
        """
        Format text with color if colors are enabled.
        
        Input: text (str), color_code (str), reset_after (bool)
        Output: str - Formatted text
        """
        if not self.use_colors:
            return text
        
        formatted = f"{color_code}{text}"
        if reset_after:
            formatted += Colors.RESET
        return formatted
    
    def create_box(self, text, box_char="═", corner_char="╔", color=Colors.SYSTEM):
        """
        Create a bordered box around text. Supports multi-line messages.
        Each line is padded to the longest line.
        """
        lines = text.split("\n")
        max_len = max(len(line) for line in lines)
        width = max_len + 4
        if not self.use_colors:
            top_border = "+" + "-" * (width - 2) + "+"
            middle = "\n".join(f"| {line.ljust(max_len)} |" for line in lines)
            bottom_border = top_border
            return f"\n{top_border}\n{middle}\n{bottom_border}\n"
        if corner_char == "╔":
            top_left, top_right = "╔", "╗"
            bottom_left, bottom_right = "╚", "╝"
            vertical = "║"
        else:
            top_left = top_right = bottom_left = bottom_right = "+"
            vertical = "|"
        top_border = f"{color}{top_left}{box_char * (width - 2)}{top_right}{Colors.RESET}"
        middle = "\n".join(f"{color}{vertical}{Colors.RESET} {line.ljust(max_len)} {color}{vertical}{Colors.RESET}" for line in lines)
        bottom_border = f"{color}{bottom_left}{box_char * (width - 2)}{bottom_right}{Colors.RESET}"
        return f"\n{top_border}\n{middle}\n{bottom_border}\n"
    
    def system_message(self, text, caps=False):
        """Format system message with 'System:' label above the box. Label is colored if colors enabled."""
        if caps:
            text = text.upper()
        label = "System:"
        if self.use_colors:
            label = Colors.BOLD + Colors.CYAN + label + Colors.RESET
        box = self.create_box(text, "═", "╔", Colors.BLUE + Colors.BOLD if self.use_colors else "")
        return f"\n\n{label}{box}"
    
    def user_prompt(self, text="User: "):
        """Format user input prompt."""
        return self.format_text(text, Colors.USER_PROMPT)
    
    def error_message(self, text, caps=False):
        """Format error message."""
        if caps:
            text = text.upper()
        if not self.use_colors:
            return f"ERROR: {text}"
        return self.create_box(f"ERROR: {text}", "═", "╔", Colors.RED + Colors.BOLD)
    
    def success_message(self, text, caps=False):
        """Format success message."""
        if caps:
            text = text.upper()
        if not self.use_colors:
            return f"SUCCESS: {text}"
        return self.create_box(f"✓ {text}", "═", "╔", Colors.GREEN + Colors.BOLD)
    
    def info_message(self, text, caps=False):
        """Format informational message. Inline ℹ style for all modes."""
        if caps:
            text = text.upper()
        return self.format_text(f"ℹ {text}", Colors.INFO)
    
    def warning_message(self, text):
        """Format warning message."""
        if not self.use_colors:
            return f"WARNING: {text}"
        
        return self.create_box(f"⚠ WARNING: {text}", "═", "╔", Colors.YELLOW + Colors.BOLD)
    
    def header(self, text, width=60, caps=False):
        """Format header text."""
        if caps:
            text = text.upper()
        if not self.use_colors:
            return f"\n{'=' * width}\n{text.center(width)}\n{'=' * width}\n"
        border = self.format_text("═" * width, Colors.HEADER)
        title = self.format_text(text.center(width), Colors.HEADER + Colors.BOLD)
        return f"\n{border}\n{title}\n{border}\n"
    
    def separator(self, width=30):
        """Format separator line."""
        if not self.use_colors:
            return "-" * width
        
        return self.format_text("─" * width, Colors.SEPARATOR)
    
    def highlight(self, text):
        """Highlight important text."""
        return self.format_text(text, Colors.BOLD + Colors.CYAN)
    
    def dim_text(self, text):
        """Dim less important text."""
        return self.format_text(text, Colors.DIM)


def create_color_formatter(use_colors=False):
    """
    Factory function to create a color formatter.
    
    Input: use_colors (bool)
    Output: ColorFormatter instance
    """
    return ColorFormatter(use_colors)