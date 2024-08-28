import functools
import inspect
import os
from loguru import logger


class deprecated(object):
    """Description:
        used to mark functions, methods and classes deprecated, and prints warning message when it called
        decorators based on https://stackoverflow.com/a/40301488

    Usage:
        todo: write usage

    Author: YingzhiGou
    Date: 20/06/2017
    """

    def __init__(self, reason):  # pragma: no cover
        if inspect.isclass(reason) or inspect.isfunction(reason):
            raise TypeError("Reason for deprecation must be supplied")
        self.reason = reason

    def __call__(self, cls_or_func):  # pragma: no cover
        """Call function."""
        if inspect.isfunction(cls_or_func):
            if hasattr(cls_or_func, "func_code"):
                _code = cls_or_func.__code__
            else:
                _code = cls_or_func.__code__
            fmt = "Call to deprecated function or method {name} ({reason})."
            filename = _code.co_filename
            lineno = _code.co_firstlineno + 1

        elif inspect.isclass(cls_or_func):
            fmt = "Call to deprecated class {name} ({reason})."
            filename = cls_or_func.__module__
            lineno = 1

        else:
            raise TypeError(type(cls_or_func))

        msg = fmt.format(name=cls_or_func.__name__, reason=self.reason)

        @functools.wraps(cls_or_func)
        def new_func(*args, **kwargs):  # pragma: no cover
            """New func."""
            import warnings

            warnings.simplefilter(
                "always", DeprecationWarning
            )  # turn off filter
            warnings.warn_explicit(
                msg,
                category=DeprecationWarning,
                filename=filename,
                lineno=lineno,
            )
            warnings.simplefilter("default", DeprecationWarning)  # reset filter
            return cls_or_func(*args, **kwargs)

        return new_func
