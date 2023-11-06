import pytest

from opensearch_py_ml.ml_commons.validators.model_access_control import (
    _validate_model_group_name,
    _validate_model_group_description,
    _validate_model_group_access_mode,
    _validate_model_group_backend_roles,
    _validate_model_group_add_all_backend_roles,
    _validate_model_group_query,
    validate_create_model_group_parameters,
    validate_update_model_group_parameters,
    validate_delete_model_group_parameters,
    validate_search_model_group_parameters
)


def test_validate_model_group_name():
    with pytest.raises(ValueError):
        _validate_model_group_name(None)

    with pytest.raises(ValueError):
        _validate_model_group_name("")

    with pytest.raises(ValueError):
        _validate_model_group_name(123)

    res = _validate_model_group_name("ValidName")
    assert res is None


def test_validate_model_group_description():
    with pytest.raises(ValueError):
        _validate_model_group_description(123)

    res = _validate_model_group_description("")
    assert res is None

    res = _validate_model_group_description(None)
    assert res is None

    res = _validate_model_group_description("ValidName")
    assert res is None


def test_validate_model_group_access_mode():
    with pytest.raises(ValueError):
        _validate_model_group_access_mode(123)
    
    res = _validate_model_group_access_mode("private")
    assert res is None
    
    res = _validate_model_group_access_mode("restricted")
    assert res is None
    
    res = _validate_model_group_access_mode(None)
    assert res is None

def test_validate_model_group_backend_roles():
    with pytest.raises(ValueError):
        _validate_model_group_backend_roles(123)
    
    res = _validate_model_group_backend_roles(["admin"])
    assert res is None
    
    res = _validate_model_group_backend_roles(None)
    assert res is None

def test_validate_model_group_add_all_backend_roles():
    with pytest.raises(ValueError):
        _validate_model_group_add_all_backend_roles(123)
    
    res = _validate_model_group_add_all_backend_roles(False)
    assert res is None
    
    res = _validate_model_group_add_all_backend_roles(True)
    assert res is None
    
    res = _validate_model_group_add_all_backend_roles(None)
    assert res is None


def test_validate_model_group_query():
    with pytest.raises(ValueError):
        _validate_model_group_query(123)
    
    res = _validate_model_group_query({})
    assert res is None
    
    with pytest.raises(ValueError):
        _validate_model_group_query(None)
    
    res = _validate_model_group_query({"query": {"match": {"name": "test"}}})
    assert res is None


def test_validate_create_model_group_parameters():
    with pytest.raises(ValueError):
        validate_create_model_group_parameters(123)
    
    res = validate_create_model_group_parameters("test")
    assert res is None
    
    with pytest.raises(ValueError):
        validate_create_model_group_parameters("test", access_mode="restricted")
    
    with pytest.raises(ValueError):
        validate_create_model_group_parameters("test", access_mode="private",add_all_backend_roles=True)
    

def test_validate_update_model_group_parameters():
    with pytest.raises(ValueError):
        validate_update_model_group_parameters(123, 123)
    
    res = validate_update_model_group_parameters({"query": {}}, "test")
    assert res is None

def test_validate_delete_model_group_parameters():
    with pytest.raises(ValueError):
        validate_delete_model_group_parameters(123)
    
    res = validate_delete_model_group_parameters("test")
    assert res is None

def test_validate_search_model_group_parameters():
    with pytest.raises(ValueError):
        validate_search_model_group_parameters(123)
    
    res = validate_search_model_group_parameters({"query": {}})
    assert res is None

    
    