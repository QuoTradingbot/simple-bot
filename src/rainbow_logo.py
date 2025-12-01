"""
Rainbow ASCII Art Logo for QuoTrading AI
Displays animated "QuoTrading AI" logo with rainbow colors that slowly transition
"""

import time
import sys
import os


# ANSI color codes for rainbow effect
class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Rainbow colors
    RED = '\033[91m'
    ORANGE = '\033[38;5;208m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    MAGENTA = '\033[35m'


# ASCII Art for "QUOTRADING AI" - Outline style that clearly spells the words
QUO_AI_LOGO = [
    "  ██████╗ ██╗   ██╗ ██████╗ ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗      █████╗ ██╗",
    " ██╔═══██╗██║   ██║██╔═══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝     ██╔══██╗██║",
    " ██║   ██║██║   ██║██║   ██║   ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗    ███████║██║",
    " ██║▄▄ ██║██║   ██║██║   ██║   ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║    ██╔══██║██║",
    " ╚██████╔╝╚██████╔╝╚██████╔╝   ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝    ██║  ██║██║",
    "  ╚═════╝  ╚═════╝  ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝     ╚═╝  ╚═╝╚═╝"
]

# Subtitle for professional branding
SUBTITLE = "A L G O R I T H M I C   T R A D I N G"


def get_rainbow_colors():
    """Get list of rainbow colors in order"""
    return [
        Colors.RED,
        Colors.ORANGE,
        Colors.YELLOW,
        Colors.GREEN,
        Colors.CYAN,
        Colors.BLUE,
        Colors.PURPLE,
        Colors.MAGENTA
    ]


def color_char_with_gradient(char, position, total_chars, color_offset=0):
    """
    Color a single character based on its position in the line.
    Creates a rainbow gradient effect across the entire line.
    
    Args:
        char: Character to color
        position: Position of character in the line
        total_chars: Total characters in the line
        color_offset: Offset to shift the rainbow (for animation)
    
    Returns:
        Colored character string
    """
    if char.strip() == '':
        return char
    
    # Handle empty lines
    if total_chars == 0:
        return char
    
    rainbow = get_rainbow_colors()
    
    # Calculate which color to use based on position
    # Divide the line into segments, one for each color
    color_index = int((position / total_chars) * len(rainbow) + color_offset) % len(rainbow)
    color = rainbow[color_index]
    
    return f"{color}{char}{Colors.RESET}"


def color_line_with_gradient(line, color_offset):
    """
    Color a line with rainbow gradient.
    
    Args:
        line: Line of text to color
        color_offset: Offset for rainbow animation
    
    Returns:
        Colored line string
    """
    total_chars = len(line)
    colored_line = ""
    for i, char in enumerate(line):
        colored_line += color_char_with_gradient(char, i, total_chars, color_offset)
    return colored_line


def color_line_with_gradient_and_fade(line, color_offset, fade_progress):
    """
    Color a line with rainbow gradient and fade-in effect.
    Starts super dark and gradually becomes more visible.
    
    Args:
        line: Line of text to color
        color_offset: Offset for rainbow animation
        fade_progress: Progress of fade (0.0 = super dark, 1.0 = fully visible)
    
    Returns:
        Colored line string with fade effect
    """
    total_chars = len(line)
    colored_line = ""
    
    # Calculate brightness based on fade progress
    # Start at 0 (invisible/super dark) and go to 100 (fully visible)
    brightness = int(fade_progress * 100)
    
    # Use dim text ANSI code for fade effect
    if fade_progress < 0.3:
        # Super dark - use dark gray color
        dim_code = '\033[2m\033[90m'  # Dim + dark gray
    elif fade_progress < 0.6:
        # Medium - use regular gray
        dim_code = '\033[90m'  # Dark gray
    elif fade_progress < 0.9:
        # Getting visible - use light gray
        dim_code = '\033[37m'  # Light gray
    else:
        # Fully visible - use rainbow colors
        dim_code = ''
    
    if fade_progress < 0.9:
        # Apply dim effect during fade-in
        for char in line:
            if char.strip():
                colored_line += dim_code + char + Colors.RESET
            else:
                colored_line += char
    else:
        # Fully visible - use rainbow gradient
        for i, char in enumerate(line):
            colored_line += color_char_with_gradient(char, i, total_chars, color_offset)
    
    return colored_line


def display_logo_line(line, color_offset=0, center_width=80):
    """
    Display a single line of the logo with rainbow gradient, centered.
    
    Args:
        line: Line of ASCII art to display
        color_offset: Offset for rainbow animation
        center_width: Width to center the text within (default: 80)
    """
    colored_line = color_line_with_gradient(line, color_offset)
    
    # Center the line
    padding = (center_width - len(line)) // 2
    print(" " * padding + colored_line)


def display_animated_logo(duration=3.0, fps=20, with_headers=True):
    """
    Display the QuoTrading AI logo with animated rainbow colors.
    Professional splash screen - shows logo with flowing rainbow gradient.
    Subtitle fades in from dark to visible over time.
    
    Args:
        duration: How long to display in seconds (default: 3.0, reduced for faster startup)
        fps: Frames per second for animation (default: 20, increased for smoother animation)
        with_headers: Whether to show header/footer text (default: True)
    """
    frames = int(duration * fps)
    delay = 1.0 / fps
    
    # Get terminal dimensions for centering
    try:
        terminal_size = os.get_terminal_size()
        terminal_width = terminal_size.columns
        terminal_height = terminal_size.lines
    except OSError:
        # Terminal size cannot be determined (e.g., output redirected)
        terminal_width = 120
        terminal_height = 30
    
    # Calculate vertical centering - center the logo in the middle of screen
    logo_lines = len(QUO_AI_LOGO) + 2  # Logo + blank + subtitle
    vertical_padding = max(0, (terminal_height - logo_lines) // 2)
    
    # We'll update the display in place using carriage return and line clearing
    # Number of lines we'll be updating
    total_display_lines = len(QUO_AI_LOGO) + 2  # Logo + blank + subtitle
    
    for frame in range(frames):
        # Calculate color offset for flowing rainbow effect
        color_offset = (frame / frames) * len(get_rainbow_colors())
        
        # Calculate fade-in progress for subtitle (0.0 to 1.0)
        fade_progress = frame / max(1, frames - 1)
        
        # If first frame, add vertical padding to center logo
        if frame == 0:
            # Add top padding for vertical centering
            print("\n" * vertical_padding, end='')
            sys.stdout.flush()
        else:
            # Move cursor up to beginning of logo (not including top padding)
            sys.stdout.write(f'\033[{total_display_lines}A')
        
        # Display each line of logo with rainbow colors
        for line in QUO_AI_LOGO:
            # Clear the line first
            sys.stdout.write('\033[2K')
            # Get colored line and center it horizontally
            colored_line = color_line_with_gradient(line, color_offset)
            padding = max(0, (terminal_width - len(line)) // 2)
            sys.stdout.write(" " * padding + colored_line + "\n")
        
        # Blank line
        sys.stdout.write('\033[2K\n')
        
        # Subtitle with fade-in effect (centered)
        sys.stdout.write('\033[2K')  # Clear line
        # Apply rainbow gradient to subtitle with fade-in
        subtitle_colored = color_line_with_gradient_and_fade(SUBTITLE, color_offset, fade_progress)
        subtitle_padding = max(0, (terminal_width - len(SUBTITLE)) // 2)
        sys.stdout.write(" " * subtitle_padding + subtitle_colored + "\n")
        
        # Flush to ensure immediate display
        sys.stdout.flush()
        
        # Wait before next frame
        if frame < frames - 1:
            time.sleep(delay)
    
    # Add spacing after logo
    if not with_headers:
        print("\n" * 2)
    else:
        print("\n" + "=" * 60)
        print(" " * 20 + "INITIALIZING...")
        print("=" * 60 + "\n")


def display_static_logo():
    """Display the logo without animation (single rainbow gradient)"""
    print()
    for line in QUO_AI_LOGO:
        display_logo_line(line, 0)
    print()


if __name__ == "__main__":
    # Test the logo display
    print("Testing QuoTrading AI Rainbow Logo...")
    print("=" * 60)
    display_animated_logo(duration=5.0, fps=15)
    print("=" * 60)
    print("Logo test complete!")
