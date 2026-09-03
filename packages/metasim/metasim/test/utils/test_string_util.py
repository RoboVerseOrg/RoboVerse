"""Tests for metasim.utils.string_util module.

These are pure unit tests for string utility functions.
All tests are marked @pytest.mark.general.
"""

import pytest

from metasim.utils.string_util import (
    callable_to_string,
    is_camel_case,
    is_snake_case,
    string_to_callable,
    to_camel_case,
    to_snake_case,
)


@pytest.mark.general
def test_is_camel_case_valid():
    """Test is_camel_case with valid camel case strings."""
    assert is_camel_case("CamelCase")
    assert is_camel_case("MyClass")
    assert is_camel_case("HTTPServer")
    assert is_camel_case("A")


@pytest.mark.general
def test_is_camel_case_invalid():
    """Test is_camel_case with invalid strings."""
    assert not is_camel_case("camelCase")  # starts with lowercase
    assert not is_camel_case("snake_case")
    assert not is_camel_case("lowercase")


@pytest.mark.general
def test_is_snake_case_valid():
    """Test is_snake_case with valid snake case strings."""
    assert is_snake_case("snake_case")
    assert is_snake_case("my_variable")
    assert is_snake_case("test123")
    assert is_snake_case("a")
    assert is_snake_case("test_123_var")


@pytest.mark.general
def test_is_snake_case_invalid():
    """Test is_snake_case with invalid strings."""
    assert not is_snake_case("CamelCase")
    assert not is_snake_case("camelCase")
    assert not is_snake_case("_private")  # starts with underscore
    assert not is_snake_case("CONSTANT")  # uppercase


@pytest.mark.general
def test_to_camel_case_simple():
    """Test to_camel_case conversion."""
    assert to_camel_case("snake_case") == "SnakeCase"
    assert to_camel_case("my_variable_name") == "MyVariableName"
    assert to_camel_case("test") == "Test"
    assert to_camel_case("a_b_c") == "ABC"


@pytest.mark.general
def test_to_snake_case_simple():
    """Test to_snake_case conversion."""
    assert to_snake_case("CamelCase") == "camel_case"
    assert to_snake_case("MyClassName") == "my_class_name"
    assert to_snake_case("HTTPServer") == "http_server"
    assert to_snake_case("A") == "a"


@pytest.mark.general
def test_camel_snake_roundtrip():
    """Test that snake->camel->snake roundtrips correctly."""
    original = "my_test_variable"
    camel = to_camel_case(original)
    back_to_snake = to_snake_case(camel)

    assert back_to_snake == original


@pytest.mark.general
def test_callable_to_string_regular_function():
    """Test callable_to_string with regular function."""

    def my_test_function():
        pass

    result = callable_to_string(my_test_function)

    # Should be in format "module:function_name"
    assert ":" in result
    assert "my_test_function" in result


@pytest.mark.general
def test_callable_to_string_builtin():
    """Test callable_to_string with builtin function."""
    result = callable_to_string(len)

    assert ":" in result
    assert "len" in result


@pytest.mark.general
def test_callable_to_string_non_callable():
    """Test callable_to_string raises ValueError for non-callable."""
    with pytest.raises(ValueError, match="not callable"):
        callable_to_string("not a function")

    with pytest.raises(ValueError, match="not callable"):
        callable_to_string(123)


@pytest.mark.general
def test_string_to_callable_builtin():
    """Test string_to_callable with builtin function."""
    result = string_to_callable("builtins:len")

    assert callable(result)
    assert result([1, 2, 3]) == 3


@pytest.mark.general
def test_string_to_callable_roundtrip():
    """Test that callable->string->callable roundtrips correctly with importable functions."""
    import math

    # Use an importable function
    # Convert to string
    func_string = callable_to_string(math.sqrt)

    # Convert back to callable
    func_restored = string_to_callable(func_string)

    # Should work the same way
    assert func_restored(25) == math.sqrt(25) == 5.0


@pytest.mark.general
def test_string_to_callable_invalid_format():
    """Test string_to_callable raises ValueError for invalid format."""
    with pytest.raises(ValueError, match="Could not resolve"):
        string_to_callable("invalid_format_no_colon")

    with pytest.raises(ValueError, match="Could not resolve"):
        string_to_callable("nonexistent_module:function")


@pytest.mark.general
def test_string_to_callable_not_callable():
    """Test string_to_callable raises error when attribute is not callable."""
    # sys.version is a string, not callable
    with pytest.raises(AttributeError, match="not callable"):
        string_to_callable("sys:version")
