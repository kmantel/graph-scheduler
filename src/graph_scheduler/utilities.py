import inspect
import logging
import weakref

logger = logging.getLogger(__name__)

_unused_args_sig_cache = weakref.WeakKeyDictionary()


def prune_unused_args(func, args=None, kwargs=None):
    """
        Arguments
        ---------
            func : function

            args : *args

            kwargs : **kwargs


        Returns
        -------
            a tuple such that the first item is the intersection of **args** and the
            positional arguments of **func**, and the second item is the intersection
            of **kwargs** and the keyword arguments of **func**

    """
    # use the func signature to filter out arguments that aren't compatible
    try:
        sig = _unused_args_sig_cache[func]
    except KeyError:
        sig = inspect.signature(func)
        _unused_args_sig_cache[func] = sig

    has_args_param = False
    has_kwargs_param = False
    count_positional = 0
    func_kwargs_names = set()

    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            has_args_param = True
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            has_kwargs_param = True
        elif param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD or param.kind is inspect.Parameter.KEYWORD_ONLY:
            if param.default is inspect.Parameter.empty:
                count_positional += 1
            func_kwargs_names.add(name)

    if args is not None:
        try:
            args = list(args)
        except TypeError:
            args = [args]

        if not has_args_param:
            num_extra_args = len(args) - count_positional
            if num_extra_args > 0:
                logger.debug('{1} extra arguments specified to function {0}, will be ignored (values: {2})'.format(func, num_extra_args, args[-num_extra_args:]))
            args = args[:count_positional]
    else:
        args = []

    if kwargs is not None:
        kwargs = dict(kwargs)

        if not has_kwargs_param:
            filtered = set()
            for kw in kwargs:
                if kw not in func_kwargs_names:
                    filtered.add(kw)
            if len(filtered) > 0:
                logger.debug('{1} extra keyword arguments specified to function {0}, will be ignored (values: {2})'.format(func, len(filtered), filtered))
            for kw in filtered:
                del kwargs[kw]
    else:
        kwargs = {}

    return args, kwargs


def call_with_pruned_args(func, *args, **kwargs):
    """
        Calls **func** with only the **args** and **kwargs** that
        exist in its signature
    """
    args, kwargs = prune_unused_args(func, args, kwargs)
    return func(*args, **kwargs)
