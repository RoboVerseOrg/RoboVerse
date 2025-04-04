# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for transforming strings and regular expressions."""

import ast
import importlib
import inspect
import re
from collections.abc import Callable
from dataclasses import is_dataclass

"""
String formatting.
"""


def is_camel_case(name: str) -> bool:
    """Check if a string is in camel case.

    Args:
        name: A string to check.

    Returns:
        Whether the string is in camel case.
    """
    # see https://stackoverflow.com/a/10182711
    return re.match(r"^(?:[A-Z0-9][a-z]*)+$", name) is not None


def is_snake_case(name: str) -> bool:
    """Check if a string is in snake case.

    Args:
        name: A string to check.

    Returns:
        Whether the string is in snake case.
    """
    return re.match(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$", name) is not None


def to_camel_case(snake_str: str) -> str:
    """Convert a string from snake case to camel case.

    Args:
        snake_str: A string in snake case.

    Returns:
        A string in camel case (i.e. with no '_').
    """
    components = snake_str.lower().split("_")
    return "".join(x.title() for x in components)


def to_snake_case(camel_str: str) -> str:
    """Convert a string from camel case to snake case.

    Args:
        camel_str: A string in camel case.

    Returns:
        A string in snake case (i.e. with '_')
    """
    camel_str = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel_str).lower()


"""
String <-> Callable operations.
"""


def is_lambda_expression(name: str) -> bool:
    """Checks if the input string is a lambda expression.

    Args:
        name: The input string.

    Returns:
        Whether the input string is a lambda expression.
    """
    try:
        ast.parse(name)
        return isinstance(ast.parse(name).body[0], ast.Expr) and isinstance(ast.parse(name).body[0].value, ast.Lambda)
    except SyntaxError:
        return False


def callable_to_string(value: Callable) -> str:
    """Converts a callable object to a string.

    Args:
        value: A callable object.

    Raises:
        ValueError: When the input argument is not a callable object.

    Returns:
        A string representation of the callable object.
    """
    # check if callable
    if not callable(value):
        raise ValueError(f"The input argument is not callable: {value}.")
    # check if lambda function
    if value.__name__ == "<lambda>":
        # we resolve the lambda expression by checking the source code and extracting the line with lambda expression
        # we also remove any comments from the line
        lambda_line = inspect.getsourcelines(value)[0][0].strip().split("lambda")[1].strip().split(",")[0]
        lambda_line = re.sub(r"#.*$", "", lambda_line).rstrip()
        return f"lambda {lambda_line}"
    else:
        # get the module and function name
        module_name = value.__module__
        function_name = value.__name__
        # return the string
        return f"{module_name}:{function_name}"


def string_to_callable(name: str) -> Callable:
    """Resolves the module and function names to return the function.

    Args:
        name: The function name. The format should be 'module:attribute_name' or a
            lambda expression of format: 'lambda x: x'.

    Raises:
        ValueError: When the resolved attribute is not a function.
        ValueError: When the module cannot be found.

    Returns:
        Callable: The function loaded from the module.
    """
    try:
        if is_lambda_expression(name):
            callable_object = eval(name)
        else:
            mod_name, attr_name = name.split(":")
            mod = importlib.import_module(mod_name)
            callable_object = getattr(mod, attr_name)
        # check if attribute is callable
        if callable(callable_object):
            return callable_object
        else:
            raise AttributeError(f"The imported object is not callable: '{name}'")
    except (ValueError, ModuleNotFoundError) as e:
        msg = (
            f"Could not resolve the input string '{name}' into callable object."
            " The format of input should be 'module:attribute_name'.\n"
            f"Received the error:\n {e}."
        )
        raise ValueError(msg) from e


def string_to_dataclass_instance(name: str) -> Callable:
    """Resolves the module and attribute name to return the dataclass instance.

    Args:
        name: The dataclass name. The format should be 'module:attribute_name'.

    Returns:
        A dataclass instance.
    """
    try:
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        dataclass_instance = getattr(mod, attr_name)
        if is_dataclass(dataclass_instance):
            if isinstance(dataclass_instance, type):
                return dataclass_instance()
            else:
                return dataclass_instance
        else:
            raise AttributeError(f"The imported object is not a dataclass: '{name}'")
    except (ValueError, ModuleNotFoundError) as e:
        msg = (
            f"Could not resolve the input string '{name}' into dataclass instance."
            " The format of input should be 'module:attribute_name'.\n"
            f"Received the error:\n {e}."
        )
        raise ValueError(msg) from e
