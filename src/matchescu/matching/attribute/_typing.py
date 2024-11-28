from typing import Callable, Any

from matchescu.matching.attribute._match import TResult


AttrMatchCallable = Callable[[Any, Any], TResult]
