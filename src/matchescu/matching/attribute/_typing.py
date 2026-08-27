from collections.abc import Callable
from typing import Any

from matchescu.matching.attribute._match import TResult

AttrMatchCallable = Callable[[Any, Any], TResult]
