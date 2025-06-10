import sys

from colorama import Style, Fore

THOUGHT_COLOR = Fore.GREEN
OBSERVATION_COLOR = Fore.YELLOW
ROUND_COLOR = Fore.BLUE
RETURN_COLOR = Fore.CYAN
CODE_COLOR = Fore.WHITE

def color_print(text, color, end="\n"):
    if color is not None:
        content = color + text + Style.RESET_ALL + end
    else:
        content = text + end
    sys.stdout.write(content)
    sys.stdout.flush()
