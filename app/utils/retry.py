import httpx
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)

_NET = (httpx.TimeoutException, httpx.ConnectError, ConnectionError, OSError)

llm_retry = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(min=2, max=30),
    retry=retry_if_exception_type(_NET)
)

embed_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=15),
    retry=retry_if_exception_type(_NET)
)
