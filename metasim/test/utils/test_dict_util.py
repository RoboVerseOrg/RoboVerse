"""Tests for metasim.utils.dict module.

These are pure unit tests for dictionary utility functions.
All tests are marked @pytest.mark.general.
"""

import pytest

from metasim.utils.dict import class_to_dict, deep_get, update_class_from_dict


class SimpleClass:
    """Simple test class."""

    def __init__(self):
        self.a = 1
        self.b = "test"
        self.c = [1, 2, 3]


class NestedClass:
    """Nested test class."""

    def __init__(self):
        self.simple = SimpleClass()
        self.value = 42


@pytest.mark.general
def test_deep_get_simple():
    """Test deep_get with simple dictionary."""
    d = {"a": {"b": {"c": 123}}}

    result = deep_get(d, "a", "b", "c")
    assert result == 123


@pytest.mark.general
def test_deep_get_missing_key():
    """Test deep_get with missing key returns None."""
    d = {"a": {"b": 123}}

    result = deep_get(d, "a", "c")
    assert result is None

    result = deep_get(d, "x", "y", "z")
    assert result is None


@pytest.mark.general
def test_deep_get_single_key():
    """Test deep_get with single key."""
    d = {"a": 123}

    result = deep_get(d, "a")
    assert result == 123


@pytest.mark.general
def test_class_to_dict_simple():
    """Test class_to_dict with simple class."""
    obj = SimpleClass()
    result = class_to_dict(obj)

    assert result["a"] == 1
    assert result["b"] == "test"
    assert result["c"] == [1, 2, 3]


@pytest.mark.general
def test_class_to_dict_nested():
    """Test class_to_dict with nested class."""
    obj = NestedClass()
    result = class_to_dict(obj)

    assert result["value"] == 42
    assert isinstance(result["simple"], dict)
    assert result["simple"]["a"] == 1
    assert result["simple"]["b"] == "test"


@pytest.mark.general
def test_class_to_dict_ignores_private():
    """Test that class_to_dict ignores private attributes."""

    class ClassWithPrivate:
        def __init__(self):
            self.public = 1
            self.__private = 2

    obj = ClassWithPrivate()
    result = class_to_dict(obj)

    assert "public" in result
    assert "__private" not in result


@pytest.mark.general
def test_update_class_from_dict_simple():
    """Test update_class_from_dict with simple values."""
    obj = SimpleClass()
    data = {"a": 99, "b": "updated"}

    update_class_from_dict(obj, data)

    assert obj.a == 99
    assert obj.b == "updated"
    assert obj.c == [1, 2, 3]  # Unchanged


@pytest.mark.general
def test_update_class_from_dict_nested():
    """Test update_class_from_dict with nested dictionaries."""
    obj = NestedClass()
    data = {"value": 100, "simple": {"a": 50}}

    update_class_from_dict(obj, data)

    assert obj.value == 100
    assert obj.simple.a == 50
    assert obj.simple.b == "test"  # Unchanged


@pytest.mark.general
def test_update_class_from_dict_invalid_key():
    """Test update_class_from_dict raises KeyError for invalid key."""
    obj = SimpleClass()
    data = {"nonexistent": 123}

    with pytest.raises(KeyError, match="Key not found"):
        update_class_from_dict(obj, data)


@pytest.mark.general
def test_update_class_from_dict_type_mismatch():
    """Test update_class_from_dict raises ValueError for type mismatch."""
    obj = SimpleClass()
    data = {"a": "not_an_int"}  # a should be int

    with pytest.raises(ValueError, match="Incorrect type"):
        update_class_from_dict(obj, data)


@pytest.mark.general
def test_update_class_from_dict_list_length_mismatch():
    """Test update_class_from_dict raises ValueError for list length mismatch."""
    obj = SimpleClass()
    data = {"c": [1, 2]}  # Original has 3 elements

    with pytest.raises(ValueError, match="Incorrect length"):
        update_class_from_dict(obj, data)


@pytest.mark.general
def test_update_class_from_dict_with_dict_object():
    """Test update_class_from_dict works with dict objects."""
    obj = {"a": 1, "b": {"c": 2}}
    data = {"a": 10, "b": {"c": 20}}

    update_class_from_dict(obj, data)

    assert obj["a"] == 10
    assert obj["b"]["c"] == 20


@pytest.mark.general
def test_update_class_from_dict_with_none():
    """Test update_class_from_dict handles None values."""
    obj = SimpleClass()
    obj.a = None  # Allow None
    data = {"a": None}

    update_class_from_dict(obj, data)

    assert obj.a is None
