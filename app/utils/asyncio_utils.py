"""
Asyncio utility functions and decorators
"""

import asyncio
import logging
from functools import wraps
from typing import Callable, TypeVar, Awaitable

logger = logging.getLogger(__name__)

T = TypeVar('T')

def run_async(func: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """
    Decorator to run async functions in a new event loop from sync context.

    Args:
        func: The async function to wrap

    Returns:
        A sync function that runs the async function in a new event loop
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()

    return wrapper


def run_async_with_existing_loop(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine in a new event loop.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class AsyncExecutor:
    """
    Context manager for running async code in sync context.
    """

    def __init__(self):
        self.loop = None

    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.loop:
            self.loop.close()

    def run(self, coro: Awaitable[T]) -> T:
        """Run a coroutine in the managed loop."""
        if not self.loop:
            raise RuntimeError("AsyncExecutor not properly initialized")
        return self.loop.run_until_complete(coro)
