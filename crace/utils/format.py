import re
import pandas as pd

from crace.utils.const import WIDTH

_ESCAPE_RE = re.compile(r'\\([ntrfvab\\])')
_ESCAPE_MAP = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "f": "\f",
    "v": "\v",
    "a": "\a",
    "b": "\b",
    "\\": "\\",
}

def _decode_escape(text: str):
    return _ESCAPE_RE.sub(lambda m: _ESCAPE_MAP[m.group(1)], text)

def _strip_ansi(s: str) -> str:
    # CSI (colors, style, etc.)
    s = re.sub(r'\x1b\[[0-9;]*m', '', s)
    # OSC 8 hyperlinks: ESC ] 8 ;; ... ESC \
    s = re.sub(r'\x1b]8;;.*?\x1b\\', '', s)
    return s

def _ansi_wrap(text: str, width: int, hanging: int, space: bool):
    """
    textwrap based on visible width (ANSI-aware)
    preserve explicit newlines
    hanging applies to wrapped lines inside each paragraph
    """
    out = []

    paras = text.split("\n")

    for i, para in enumerate(paras):
        if para.strip() == "":
            out.append("")
            continue

        out.extend(_ansi_wrap_one_para(para, width, hanging, space, i==0))

    return out

def _ansi_wrap_one_para(text: str, width: int, hanging: int, space: bool, init_line: bool):
    base_indent = ""

    if space:
        text = text.replace(r"\t", "\t").expandtabs(2)
        m = re.match(r"^ *", text)
        base_indent = m.group(0)
        text = text[len(base_indent):]

    words = text.split()
    lines = []
    current = ""
    first = True

    available_width = width - len(base_indent) - hanging    # available width
    hanging_indent = base_indent + " " * hanging


    for w in words:

        test = w if not current else current + " " + w

        if len(_strip_ansi(test)) > available_width:
            if current:
                if first and init_line:
                    lines.append(base_indent + current)
                else:
                    lines.append(hanging_indent + current)
            current = w
            first = False
        else:
            current = test 

    if current:
        if first and init_line:
            lines.append(base_indent + current)
        else:
            lines.append(hanging_indent + current)

    return lines

def _make_ascii_box(text: str, width: int, hanging: int):
    """replace xwarningbox to ascii format"""
    # textwrap
    lines = _ansi_wrap(text, width-2, hanging)
    maxw = max(len(_strip_ansi(l)) for l in lines) if lines else 0

    top = "  +" + "-" * (maxw + 2) + "+"
    box = [top]
    for l in lines:
        pad = maxw - len(_strip_ansi(l))
        box.append(f"  | {l}{' ' * pad} |")
    box.append(top)

    return "\n".join(box)

def _make_ascii_list(text: str, width: int, hanging: int):
    """
    transform itemsize to ascii format
    """
    ITEM_RE = re.compile(r"\\item\[\]\s*(.*?)(?=\\item\[\]|$)", re.DOTALL)

    items = ITEM_RE.findall(text)
    if not items:
        return ""

    output = []
    for item in items:
        item = item.strip()

        wrapped = _ansi_wrap(item, width-4, hanging)   # with prefix "- "
        if wrapped:
            # first line with prefix "- "
            output.append(f"  - {wrapped[0]}")
            # other lines
            for w in wrapped[1:]:
                output.append(f"    {w}")
        else:
            output.append("-")

    return "\n".join(output)

def format_string(text: str, width: int=WIDTH, hanging: int=0, space: bool=False):
    """
    - transform xwarningbox → ASCII frame
    - transform itemize → ASCII list
    - textwrap for other text

    :param text: input string
    :param width: max width of output string
    :param hanging: hanging indent for wrapped lines
    :param space: whether to preserve spaces and tabs at the beginning of each line
    """
    output = []
    pos = 0

    # search patterns: warningbox -> itemize
    for match in re.finditer(
        r"(\\begin{xwarningbox}.*?\\end{xwarningbox}|"
        r"\\begin{itemize}.*?\\end{itemize})",
        text,
        flags=re.DOTALL,
    ):
        start, end = match.span()
        before = text[pos:start]

        # normal text in front
        if before.strip():
            output.append("\n".join(_ansi_wrap(before, width, hanging)))

        block = match.group(1)

        # xwarningbox
        if block.startswith(r"\begin{xwarningbox}"):
            content = re.search(r"\\begin{xwarningbox}\s*(.*?)\s*\\end{xwarningbox}", block, re.DOTALL).group(1)
            output.append(_make_ascii_box(content.strip(), width, hanging))

        # itemize
        elif block.startswith(r"\begin{itemize}"):
            content = re.search(r"\\begin{itemize}\s*(.*?)\s*\\end{itemize}", block, re.DOTALL).group(1)
            output.append(_make_ascii_list(content.strip(), width, hanging))

        pos = end

    # normal text in the end
    tail = text[pos:]
    if tail.strip():
        output.append("\n".join(_ansi_wrap(tail, width, hanging, space)))

    return "\n".join(output)


class ConditionalReturn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs if len(kwargs) != 0 else None

    def __repr__(self):
        if self.kwargs:
            for k,v in self.kwargs.items():
                if isinstance(v, pd.DataFrame) and 'title' in v.attrs and v.attrs['title']:
                    print(f"{v.attrs['title']} \n{v}\n")
                elif isinstance(v, str):
                    print(f"{v}")
                else: print(f'{k}: {v}')
        return ""
