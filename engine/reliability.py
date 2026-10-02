from dataclasses import dataclass
import time


@dataclass(frozen=True)
class AttemptResult:
    value: object
    attempts: int
    reason: str


def run_with_retries(operation, attempts=2, delay=0.2, is_success=None):
    """Run a bounded provider operation and return the first usable result."""
    limit = max(1, int(attempts))
    checker = is_success or (lambda value: value is not None)
    last_reason = "empty_result"

    for index in range(limit):
        try:
            value = operation()
        except TimeoutError:
            value = None
            last_reason = "timeout"
        except Exception:
            value = None
            last_reason = "provider_error"

        try:
            usable = bool(checker(value))
        except Exception:
            usable = False

        if usable:
            return AttemptResult(value, index + 1, "ok")

        if index + 1 < limit and delay > 0:
            time.sleep(delay)

    return AttemptResult(None, limit, last_reason)
