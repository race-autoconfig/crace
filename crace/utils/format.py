import re
import pandas as pd


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

    for para in paras:
        if para.strip() == "":
            out.append("")
            continue

        out.extend(_ansi_wrap_one_para(para, width, hanging, space))

    return out

def _ansi_wrap_one_para(text: str, width: int, hanging: int, space: bool):
    base_indent = ""

    if space:
        n = len(text) - len(text.lstrip(" "))
        base_indent = " " * n
        text = text.lstrip(" ")

    words = text.split()
    lines = []
    current = ""
    first = False

    for w in words:
        indent = "" if first else " " * hanging
        test = w if not current else current + " " + w

        if len(_strip_ansi(base_indent + indent + test)) > width:
            lines.append(base_indent + indent + current)
            current = w
            first = False
        else:
            current = test

    if current:
        indent = "" if first else " " * hanging
        lines.append(base_indent + indent + current)

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

def format_string(vignettes: str, width: int = 85, hanging: int=2, space: bool=False):
    """
    - transform xwarningbox → ASCII frame
    - transform itemize → ASCII list
    - textwrap for other text
    """
    output = []
    pos = 0

    # search patterns: warningbox -> itemize
    for match in re.finditer(
        r"(\\begin{xwarningbox}.*?\\end{xwarningbox}|"
        r"\\begin{itemize}.*?\\end{itemize})",
        vignettes,
        flags=re.DOTALL,
    ):
        start, end = match.span()
        before = vignettes[pos:start]

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
    tail = vignettes[pos:]
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
