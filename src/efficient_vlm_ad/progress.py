from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def progress(iterable: Iterable[T], *, desc: str, total: int | None = None, disable: bool = False) -> Iterator[T]:
    yield from tqdm(iterable, desc=desc, total=total, disable=disable)


def progress_bar(iterable: Iterable[T], *, desc: str, total: int | None = None, disable: bool = False):
    return tqdm(iterable, desc=desc, total=total, disable=disable)


def progress_write(message: str) -> None:
    tqdm.write(message)
    sys.stdout.flush()
    sys.stderr.flush()
