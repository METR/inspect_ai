ANSWER_PATTERN_SUFFIX = r"(?:\s*\n)*\s*$"
"""Pattern that matches only trailing blank lines and whitespace."""

ANSWER_PATTERN_LETTER = (
    r"(?im)^\s*ANSWER\s*:\s*([A-Za-z])[^\w\n]*" + ANSWER_PATTERN_SUFFIX
)
"""Match a single letter answer on the last non-blank line."""

ANSWER_PATTERN_WORD = r"(?im)^\s*ANSWER\s*:\s*(\w+)[^\w\n]*" + ANSWER_PATTERN_SUFFIX
"""Match a single word answer on the last non-blank line."""

ANSWER_PATTERN_LINE = r"(?im)^\s*ANSWER\s*:\s*(.+?)\s*" + ANSWER_PATTERN_SUFFIX
"""Match the remainder of the last non-blank line after ANSWER:."""
