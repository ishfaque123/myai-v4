import re

def calculate(text):
    text = text.lower()
    text = text.replace("plus", "+").replace("minus", "-")
    text = text.replace("times", "*").replace("multiply", "*")
    text = text.replace("divided by", "/")
    match = re.search(r"(?<!\w)(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)(?!\w)", text)
    if not match:
        return None
    a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
    if op == "+": result = a + b
    elif op == "-": result = a - b
    elif op == "*": result = a * b
    elif op == "/":
        if b == 0: return "Zero se divide nahi kar sakte."
        result = a / b
    else:
        return None
    return str(int(result)) if result.is_integer() else str(result)
