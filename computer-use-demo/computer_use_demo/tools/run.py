"""Utility to run async operations and shell commands with timeout."""

import asyncio
from typing import Any, Callable, TypeVar, Union, Tuple

T = TypeVar('T')

TRUNCATED_MESSAGE: str = "<response clipped><NOTE>Response was truncated to save context.</NOTE>"
MAX_RESPONSE_LEN: int = 16000


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN) -> str:
    """Truncate content and append a notice if content exceeds the specified length."""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )


async def run_shell(
    cmd: str,
    timeout: float | None = 120.0,  # seconds
    truncate_after: int | None = MAX_RESPONSE_LEN,
) -> Tuple[int, str, str]:
    """Run a shell command asynchronously with a timeout."""
    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return (
            process.returncode or 0,
            maybe_truncate(stdout.decode(), truncate_after=truncate_after),
            maybe_truncate(stderr.decode(), truncate_after=truncate_after),
        )
    except asyncio.TimeoutError as exc:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        raise TimeoutError(
            f"Command '{cmd}' timed out after {timeout} seconds"
        ) from exc


async def run(
    operation: Union[str, Callable[..., T]],
    *args: Any,
    timeout: float | None = 120.0,  # seconds
    truncate_after: int | None = MAX_RESPONSE_LEN,
    **kwargs: Any,
) -> Union[Tuple[int, str, str], T]:
    """
    Run an operation with a timeout. The operation can be either:
    - A shell command (string)
    - A callable (sync or async function)
    """
    if isinstance(operation, str):
        return await run_shell(operation, timeout=timeout, truncate_after=truncate_after)
    
    try:
        result = await asyncio.wait_for(
            operation(*args, **kwargs) if asyncio.iscoroutine(operation) else asyncio.to_thread(operation, *args, **kwargs),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Operation timed out after {timeout} seconds"
        ) from exc
