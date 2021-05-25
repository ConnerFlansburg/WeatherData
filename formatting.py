
import sys

SYSOUT = sys.stdout


def printWarn(message: str) -> str:
    """
    printWarn is used for coloring warnings yellow.

    :param message: The message to be colored.
    :type message: str

    :rtype: str
    """
    return f"\033[33m {message}\033[00m"


def printSuccess(message: str) -> str:
    """ Colors a string green & returns it."""
    return f"\033[32;1m{message}\033[00m"


def printError(message: str) -> None:
    """
    printError is used for coloring error messages red.

    :param message: The message to be printed.
    :type message: str

    :return: printError does not return, but rather prints to the console.
    :rtype: None
    """
    print("\033[91;1m {}\033[00m".format(message))


def printPercentage(decimalScore: float) -> str:
    """ Prints the passed decimal as a percentage & colors it based on it's value. """

    if decimalScore > 0.75:  # > 75 print in green
        return f'\033[32;1m{round(decimalScore*100,2)}%\033[00m'

    elif 0.45 < decimalScore < 0.75:  # > 45 and < 75 print yellow
        return f'\033[33;1m{round(decimalScore*100,2)}%\033[00m'

    elif decimalScore < 0.45:  # < 45 print in red
        return f'\033[91;1m{round(decimalScore*100,2)}%\033[00m'

    else:  # don't add color, but print accuracy
        return f'{decimalScore}'


def colorDecimal(decimalScore: float) -> str:
    """ Colors a decimal based on it's value & then returns it"""

    if decimalScore > 0.75:  # > 75 print in green
        return f'\033[32;1m{decimalScore}\033[00m'

    elif 0.45 < decimalScore < 0.75:  # > 45 and < 75 print yellow
        return f'\033[33;1m{decimalScore}\033[00m'

    elif decimalScore < 0.45:  # < 45 print in red
        return f'\033[91;1m{decimalScore}\033[00m'

    else:  # don't add color, but print accuracy
        return f'{decimalScore}'


def printUnderline(message: str) -> str:
    """ Underlines the passed string & then returns it. """
    return f'\x1b[4m{message}\x1b[24m'
