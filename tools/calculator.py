import re

DATE_PATTERNS = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
    re.compile(r"\b\d{2}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{4}-\d{2}\b"),
    re.compile(r"\b\d{2}-\d{4}\b"),
)

NUMBER = r"\d+(?:\.\d+)?"
TOKEN = rf"(?:{NUMBER}|[+*/-]|\(|\))"
WORD_OPS = re.compile(r"\b(?:plus|minus|times|multiply|multiplied|divided)\b", re.I)


def _blank_dates(text):
    for pattern in DATE_PATTERNS:
        text = pattern.sub(lambda match: " " * len(match.group(0)), text)
    return text


def _tokenize(text):
    tokens = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        match = re.match(rf"{NUMBER}|[+*/-]", text[pos:])
        if not match:
            return None
        token = match.group(0)
        tokens.append(token)
        pos += len(token)
    return tokens


def _evaluate(tokens):
    if not tokens:
        return None
    if tokens[0] == "-" or tokens[-1] in "+-*/":
        return None
    expect_number = True
    values = []
    operators = []
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}

    def apply_top():
        if len(values) < 2 or not operators:
            return False
        b = values.pop()
        a = values.pop()
        op = operators.pop()
        if op == "/" and b == 0:
            raise ZeroDivisionError
        values.append({"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op])
        return True

    for token in tokens:
        if re.fullmatch(NUMBER, token):
            if not expect_number:
                return None
            values.append(float(token))
            expect_number = False
        else:
            if expect_number:
                return None
            while operators and precedence[operators[-1]] >= precedence[token]:
                if not apply_top():
                    return None
            operators.append(token)
            expect_number = True
    while operators:
        if not apply_top():
            return None
    return values[0] if len(values) == 1 else None


def calculate(text):
    original = str(text or "")
    working = _blank_dates(original.lower())
    working = re.sub(r"\bmultipl(?:y|ied)\s+by\b", "*", working)
    working = re.sub(r"\bdivided\s+by\b", "/", working)
    working = WORD_OPS.sub(lambda match: {"plus": "+", "minus": "-", "times": "*", "multiply": "*", "multiplied": "*", "divided": "/"}[match.group(0).lower()], working)
    working = re.sub(r"(?<=\d)\.$", "", working.strip())

    match = re.search(rf"(?<![\w,.]){TOKEN}(?:\s*{TOKEN})*(?![\w,])", working)
    if not match:
        return None
    before = working[:match.start()].rstrip()
    after = working[match.end():].lstrip()
    if re.search(r"\d,\d", original) or (after and after[0] in "+-*/") or (before and before[-1] in "+-*/"):
        return None
    expression = match.group(0)
    if re.search(r"\d,\d", expression):
        return None
    if re.search(r"\d[.]\s*$", expression):
        expression = expression[:-1]
    tokens = _tokenize(expression)
    if tokens is None:
        return None
    try:
        result = _evaluate(tokens)
    except ZeroDivisionError:
        return "Zero se divide nahi kar sakte."
    if result is None:
        return None
    return str(int(result)) if float(result).is_integer() else str(result)
