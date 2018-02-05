# Copyright (C) 2017-2018 Michael Freitag, Shahin Amiriparian, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller
#
# This file is part of auDeep.
#
# auDeep is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# auDeep is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep. If not, see <http://www.gnu.org/licenses/>.

"""Various decorators"""
import functools
import inspect

import tensorflow as tf


def lazy_property(fun):
    """
    Defines a cached read-only property.
    
    Useful for compound properties which are constant, but expensive to calculate. The @property decorator is 
    automatically applied to the decorated function.
    
    Parameters
    ----------
    fun: method
        The method to decorate

    Returns
    -------
    method
        The decorated method
    """
    attribute = '_cache_' + fun.__name__

    @property
    @functools.wraps(fun)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, fun(self))
        return getattr(self, attribute)

    return decorator


def scoped_subgraph_initializers(klass):
    """
    Decorator which creates initializers and properties for @scoped_subgraph annotated methods.
    
    For each @scoped_subgraph decorated method with name <NAME>, this decorator performs the following steps. An 
    initialization method named init_<NAME> is created, which calls the decorated method and caches the return value
    in a field named _cache_<NAME>. The @scoped_subgraph method itself is renamed to _<NAME>_initializer, and a 
    property with name <NAME> is added, returning the cached value
    
    Parameters
    ----------
    klass: class
        The class to decorate
    Returns
    -------
    class
        The decorated class
    """
    if not inspect.isclass(klass):
        raise AttributeError("scoped_subgraph_initializers decorator may only be used on classes")

    original_methods = klass.__dict__.copy()

    for name, method in original_methods.items():
        if hasattr(method, "_scoped_subgraph"):
            def init_fn(fn_self, method=method):
                attribute = "_cache_" + method.__name__

                if not hasattr(fn_self, attribute):
                    with tf.variable_scope(method.scope or method.__name__, *method.scope_args,
                                           **method.scope_kwargs) as scope:
                        setattr(fn_self, attribute, method(fn_self))
                        setattr(fn_self, (method.scope or method.__name__) + "_scope", scope)

            @property
            @functools.wraps(method)
            def property_fn(fn_self, method=method):
                attribute = "_cache_" + method.__name__

                if not hasattr(fn_self, attribute):
                    raise AttributeError(
                        "scoped subgraph defined by function %s has not been initialized yet" % method.__name__)

                return getattr(fn_self, attribute)

            setattr(klass, "init_" + name, init_fn)
            setattr(klass, "_" + name + "_initializer", method)
            setattr(klass, name, property_fn)

    return klass


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without parentheses if no arguments are provided. 
    
    All arguments must be optional.
    
    Parameters
    ----------
    function: method
        The method to decorate

    Returns
    -------
    method
        The decorated method
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function()(args[0])
        else:
            return function(*args, **kwargs)

    return decorator


@doublewrap
def scoped_subgraph(scope: str = None, *scope_args, **scope_kwargs):
    """
    A decorator for methods which specify subgraphs of a Tensorflow graph. The code of the decorated method will only be
    executed once, so that operations are not added to the graph multiple times. Furthermore, the operations are 
    enclosed in a Tensorflow variable_scope with the arguments passed to the decorator.
    
    This decorator should be used in conjunction with the scoped_subgraph_initializers decorator, which takes care of
    creating initialization methods for the subgraphs.
    
    Parameters
    ----------
    scope: str
        The name of the scope. Defaults to the name of the decorated method
    scope_args: positional arguments
        Positional arguments to pass to variable_scope
    scope_kwargs: keyword arguments
        Keyword arguments to pass to variable_scope
        
    Returns
    -------
    method
        A method which can be used to decorate methods
    """

    def wrap_function(fn):
        fn._scoped_subgraph = True
        fn.scope = scope
        fn.scope_args = scope_args
        fn.scope_kwargs = scope_kwargs

        return fn

    return wrap_function
