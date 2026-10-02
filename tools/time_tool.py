from datetime import datetime, timedelta, timezone
import re


PAKISTAN_TZ = timezone(timedelta(hours=5))


def current_datetime():
    return datetime.now(PAKISTAN_TZ)


def answer_time_query(message, style="English"):
    text = str(message or "").lower()
    now = current_datetime()

    wants_time = bool(re.search(r"\b(time|waqt|clock|kitna time|what time)\b", text))
    wants_date = bool(re.search(r"\b(date|today|aaj|day|din|year|month|saal|tareekh)\b", text))

    if wants_time and wants_date:
        if style == "Roman Urdu/Hinglish":
            return f"Aaj {now.strftime('%d %B %Y')} hai aur waqt {now.strftime('%I:%M %p')} hai."
        return f"Today is {now.strftime('%d %B %Y')} and the time is {now.strftime('%I:%M %p')}."

    if wants_time:
        if style == "Roman Urdu/Hinglish":
            return f"Abhi waqt {now.strftime('%I:%M %p')} hai."
        return f"The current time is {now.strftime('%I:%M %p')}."

    if style == "Roman Urdu/Hinglish":
        return f"Aaj {now.strftime('%d %B %Y')} hai."
    return f"Today is {now.strftime('%d %B %Y')}."
