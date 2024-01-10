from threading import local
from typing import Iterator, Callable, Deque, Generic, TypeVar
import collections
import contextlib

T = TypeVar("T")
U = TypeVar("U")

MISSING = object()

class Stack(Generic[T]):
  """Stack supporting push/pop/peek."""

  def __init__(self):
    self._storage: Deque[T] = collections.deque()

  def __len__(self) -> int:
    return len(self._storage)

  def __iter__(self) -> Iterator[T]:
    return iter(reversed(self._storage))

  def clone(self):
    return self.map(lambda v: v)

  def map(self, fn: Callable[[T], U]) -> "Stack[U]":
    s = type(self)()
    for item in self._storage:
      s.push(fn(item))
    return s

  def pushleft(self, elem: T):
    self._storage.appendleft(elem)

  def push(self, elem: T):
    self._storage.append(elem)

  def popleft(self) -> T:
    return self._storage.popleft()

  def pop(self) -> T:
    return self._storage.pop()

  def peek(self, depth=-1, default=MISSING) -> T:
    if default is not MISSING:
      return self._storage[depth] if len(self._storage) >= -depth else default
    return self._storage[depth]

  @contextlib.contextmanager
  def __call__(self, elem: T) -> Iterator[None]:  # pytype: disable=invalid-annotation
    self.push(elem)
    try:
      yield
    finally:
      assert self.pop() is elem


class ThreadLocalStack(Stack[T], local):
  """Thread-local stack."""