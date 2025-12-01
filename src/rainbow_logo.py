"""
Rainbow ASCII Art Logo for QuoTrading AI
Displays animated "QUO AI" logo with rainbow colors that slowly transition
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


# ASCII Art for "QUO AI"
QUO_AI_LOGO = [
    "  ██████╗ ██╗   ██╗ ██████╗      █████╗ ██╗",
    " ██╔═══██╗██║   ██║██╔═══██╗    ██╔══██╗██║",
    " ██║   ██║██║   ██║██║   ██║    ███████║██║",
    " ██║▄▄ ██║██║   ██║██║   ██║    ██╔══██║██║",
    " ╚██████╔╝╚██████╔╝╚██████╔╝    ██║  ██║██║",
    "  ╚══▀▀═╝  ╚═════╝  ╚═════╝     ╚═╝  ╚═╝╚═╝"
]


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
    
    rainbow = get_rainbow_colors()
    
    # Calculate which color to use based on position
    # Divide the line into segments, one for each color
    color_index = int((position / total_chars) * len(rainbow) + color_offset) % len(rainbow)
    color = rainbow[color_index]
    
    return f"{color}{char}{Colors.RESET}"


def display_logo_line(line, color_offset=0):
    """
    Display a single line of the logo with rainbow gradient.
    
    Args:
        line: Line of ASCII art to display
        color_offset: Offset for rainbow animation
    """
    total_chars = len(line)
    colored_line = ""
    
    for i, char in enumerate(line):
        colored_line += color_char_with_gradient(char, i, total_chars, color_offset)
    
    print(colored_line)


def display_animated_logo(duration=3.0, fps=10):
    """
    Display the QUO AI logo with animated rainbow colors.
    Colors slowly transition across the logo.
    
    Args:
        duration: How long to animate in seconds
        fps: Frames per second for animation
    """
    # Don't clear screen - just display the logo with animation
    frames = int(duration * fps)
    delay = 1.0 / fps
    
    # Print a header before the logo
    print("\n" + "=" * 60)
    print(" " * 15 + "QUOTRADING AI - STARTING UP")
    print("=" * 60 + "\n")
    
    for frame in range(frames):
        # Move cursor up to overwrite previous frame (only after first frame)
        if frame > 0:
            # Move cursor up by number of logo lines
            for _ in range(len(QUO_AI_LOGO)):
                print("\033[F", end='')  # Move up one line
                print("\033[K", end='')  # Clear line
        
        # Calculate color offset for this frame
        # This creates the "flowing" rainbow effect
        color_offset = (frame / frames) * len(get_rainbow_colors())
        
        # Display logo
        for line in QUO_AI_LOGO:
            display_logo_line(line, color_offset)
        
        # Flush to ensure immediate display
        sys.stdout.flush()
        
        # Wait before next frame
        if frame < frames - 1:  # Don't sleep on last frame
            time.sleep(delay)
    
    # Print footer after logo
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
    print("Testing QUO AI Rainbow Logo...")
    print("=" * 60)
    display_animated_logo(duration=5.0, fps=15)
    print("=" * 60)
    print("Logo test complete!")
