import json
import urllib.parse
import urllib.request

SEARCH_URL = "https://api.duckduckgo.com/"

_MAX_RESULTS = 5
_MAX_TEXT = 500
_TIMEOUT = 8


def web_search(query, max_results=_MAX_RESULTS, timeout=_TIMEOUT):
    text = str(query or "").strip()
    if not text:
        return []

    try:
        limit = max(1, min(int(max_results), _MAX_RESULTS))
    except (TypeError, ValueError):
        limit = _MAX_RESULTS

    params = urllib.parse.urlencode(
        {
            "q": text,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
    )
    request = urllib.request.Request(
        f"{SEARCH_URL}?{params}",
        headers={"User-Agent": "Nivora-AI/1.0"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return []

    results = []

    abstract = data.get("AbstractText")
    abstract_url = data.get("AbstractURL")
    abstract_source = data.get("AbstractSource")
    if abstract:
        results.append(
            {
                "title": data.get("Heading") or text,
                "snippet": str(abstract)[:_MAX_TEXT],
                "url": abstract_url or "",
                "source": abstract_source or "DuckDuckGo",
            }
        )

    def add_topic(topic):
        if len(results) >= limit:
            return
        if not isinstance(topic, dict):
            return

        if "Topics" in topic:
            for child in topic.get("Topics") or []:
                add_topic(child)
                if len(results) >= limit:
                    break
            return

        snippet = topic.get("Text")
        url = topic.get("FirstURL")
        if not snippet or not url:
            return

        results.append(
            {
                "title": snippet.split(" - ", 1)[0][:120],
                "snippet": str(snippet)[:_MAX_TEXT],
                "url": str(url),
                "source": "DuckDuckGo",
            }
        )

    for topic in data.get("RelatedTopics") or []:
        add_topic(topic)
        if len(results) >= limit:
            break

    return results[:limit]


def format_web_search(query, max_results=_MAX_RESULTS, timeout=_TIMEOUT):
    results = web_search(query, max_results=max_results, timeout=timeout)
    if not results:
        return "No web results were found."

    lines = [f"Web search results for: {query}"]
    for index, result in enumerate(results, 1):
        lines.append(
            f"{index}. {result['title']}\n"
            f"   {result['snippet']}\n"
            f"   Source: {result['source']}\n"
            f"   URL: {result['url']}"
        )
    return "\n".join(lines)
