
# app/monitoring.py

import threading
import time
from collections import deque
from functools import wraps


_lock = threading.Lock()

_metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_latency_ms": 0.0,
    "total_retrieved_chunks": 0,
    "error_counts": {},
}

_recent_requests = deque(maxlen=100)


def track_request(func):
    """Track request latency, success, errors, and retrieved chunks."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        success = False
        error_type = None
        retrieved_chunks = 0

        try:
            result = func(*args, **kwargs)
            success = True

            if isinstance(result, dict):
                retrieved_chunks = len(result.get("sources", []))

            return result

        except Exception as exc:
            error_type = type(exc).__name__
            raise

        finally:
            latency_ms = (time.perf_counter() - start) * 1000

            with _lock:
                _metrics["total_requests"] += 1
                _metrics["total_latency_ms"] += latency_ms
                _metrics["total_retrieved_chunks"] += retrieved_chunks

                if success:
                    _metrics["successful_requests"] += 1
                else:
                    _metrics["failed_requests"] += 1

                    _metrics["error_counts"][error_type] = (
                        _metrics["error_counts"].get(error_type, 0) + 1
                    )

                _recent_requests.append({
                    "latency_ms": round(latency_ms, 2),
                    "success": success,
                    "error_type": error_type,
                    "retrieved_chunks": retrieved_chunks,
                    "timestamp": time.time(),
                })

    return wrapper


def get_metrics() -> dict:
    """Return a thread-safe snapshot of aggregate metrics."""
    with _lock:
        total = _metrics["total_requests"]

        return {
            **_metrics,
            "error_counts": dict(_metrics["error_counts"]),
            "average_latency_ms": (
                round(_metrics["total_latency_ms"] / total, 2)
                if total else 0.0
            ),
            "average_retrieved_chunks": (
                round(_metrics["total_retrieved_chunks"] / total, 2)
                if total else 0.0
            ),
            "recent_requests": list(_recent_requests),
        }


def reset_metrics() -> None:
    """Reset development metrics."""
    with _lock:
        _metrics.update({
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_latency_ms": 0.0,
            "total_retrieved_chunks": 0,
            "error_counts": {},
        })
        _recent_requests.clear()
