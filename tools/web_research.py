import html
import re
import urllib.error
import urllib.parse
import urllib.request

from tools.web_search import web_search

_MAX_PAGE_CHARS = 5000
_TIMEOUT = 8


def fetch_webpage(url, timeout=_TIMEOUT):
    value = str(url or "").strip()
    if not value.startswith(("http://", "https://")):
        return ""
    request = urllib.request.Request(
        value,
        headers={"User-Agent": "Nivora-AI/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(_MAX_PAGE_CHARS * 3)
    except (urllib.error.URLError, TimeoutError, OSError):
        return ""

    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"(?is)<script.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    text = re.sub(r"(?is)<noscript.*?</noscript>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_PAGE_CHARS]


def research_web(query, max_results=3, timeout=_TIMEOUT):
    results = web_search(query, max_results=max_results, timeout=timeout)
    if not results:
        return "No web research results were found."

    lines = [f"Web research for: {query}"]
    for index, result in enumerate(results[:max_results], 1):
        url = result.get("url", "")
        page = fetch_webpage(url, timeout=timeout) if url else ""
        snippet = page[:1800] if page else result.get("snippet", "")
        if not snippet:
            continue
        lines.append(
            f"{index}. {result.get('title', 'Untitled')}\n"
            f"   {snippet}\n"
            f"   Source: {result.get('source', 'Web')}\n"
            f"   URL: {url}"
        )

    return "\n".join(lines) if len(lines) > 1 else "No readable web pages were found."


def format_research_query(query):
    value = str(query or "").strip()
    return urllib.parse.quote_plus(value)
