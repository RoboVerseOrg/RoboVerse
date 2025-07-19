"""API conformance tests for all simulator handlers.

These tests ensure that ALL simulator implementations follow the exact same API
as defined in BaseSimHandler and exemplified by IsaacLabHandler.
"""

import inspect
from typing import Any, Dict, List, Type, get_type_hints
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from metasim.cfg.scenario import ScenarioCfg
from metasim.sim.base import BaseSimHandler
from metasim.types import (Action, EnvState, Extra, Obs, Reward, Success,
                           TimeOut)
from metasim.utils.state import TensorState


class TestSimulatorAPIConformance:
    """Test suite to ensure all simulators conform to the same API."""

    @pytest.fixture(scope="class")
    def reference_handler_class(self):
        """Get the reference handler class (IsaacLab)."""
        try:
            from metasim.sim.isaaclab.isaaclab import IsaaclabHandler
            return IsaaclabHandler
        except ImportError:
            pytest.skip("IsaacLab reference implementation not available")

    @pytest.fixture(scope="class")
    def all_handler_classes(self) -> List[Type[BaseSimHandler]]:
        """Collect all simulator handler classes."""
        handlers = []

        handler_imports = [
            ("metasim.sim.isaaclab.isaaclab", "IsaaclabHandler"),
            ("metasim.sim.mujoco.mujoco", "MujocoHandler"),
            ("metasim.sim.genesis.genesis", "GenesisHandler"),
            ("metasim.sim.isaacgym.isaacgym", "IsaacGymHandler"),
            ("metasim.sim.sapien.sapien2", "Sapien2Handler"),
            ("metasim.sim.sapien.sapien3", "Sapien3Handler"),
            ("metasim.sim.pybullet.pybullet", "PyBulletHandler"),
            ("metasim.sim.mjx.mjx", "MJXHandler"),
            ("metasim.sim.pyrep.pyrep", "PyRepHandler"),
            ("metasim.sim.blender.blender", "BlenderHandler"),
        ]

        for module_path, class_name in handler_imports:
            try:
                module = __import__(module_path, fromlist=[class_name])
                handler_class = getattr(module, class_name)
                if issubclass(handler_class, BaseSimHandler):
                    handlers.append(handler_class)
            except (ImportError, AttributeError) as e:
                pass

        if not handlers:
            pytest.skip("No simulator handlers available for testing")

        return handlers

    @pytest.fixture(scope="class")
    def reference_signatures(self, reference_handler_class) -> Dict[str, inspect.Signature]:
        """Extract method signatures from the reference implementation."""
        signatures = {}

        for name, method in inspect.getmembers(reference_handler_class, inspect.isfunction):
            if not name.startswith('_') and name != '__init__':
                signatures[name] = inspect.signature(method)

        for name, prop in inspect.getmembers(reference_handler_class, lambda x: isinstance(x, property)):
            if not name.startswith('_'):
                signatures[name] = inspect.Signature()

        return signatures

    def test_all_handlers_inherit_from_base(self, all_handler_classes):
        """Verify all handlers inherit from BaseSimHandler."""
        for handler_class in all_handler_classes:
            assert issubclass(handler_class, BaseSimHandler), \
                f"{handler_class.__name__} must inherit from BaseSimHandler"

    def test_required_methods_exist(self, all_handler_classes):
        """Test that all required methods from BaseSimHandler exist."""
        required_methods = [
            'launch', 'step', 'reset', 'render', 'close',
            '_set_states', '_get_states', '_simulate',
            'set_dof_targets', 'get_joint_names', 'get_body_names',
            'refresh_render'
        ]

        for handler_class in all_handler_classes:
            for method_name in required_methods:
                assert hasattr(handler_class, method_name), \
                    f"{handler_class.__name__} missing required method: {method_name}"

                method = getattr(handler_class, method_name)
                assert callable(method), \
                    f"{handler_class.__name__}.{method_name} must be callable"

    def test_required_properties_exist(self, all_handler_classes):
        """Test that all required properties exist."""
        required_properties = [
            'episode_length_buf', 'actions_cache', 'device'
        ]

        for handler_class in all_handler_classes:
            for prop_name in required_properties:
                assert hasattr(handler_class, prop_name), \
                    f"{handler_class.__name__} missing required property: {prop_name}"

                prop = getattr(handler_class, prop_name)
                assert isinstance(prop, property), \
                    f"{handler_class.__name__}.{prop_name} must be a property"

    def test_method_signatures_match_reference(self, all_handler_classes, reference_signatures):
        """Test that method signatures match the reference implementation."""
        for handler_class in all_handler_classes:
            handler_name = handler_class.__name__

            for method_name, ref_sig in reference_signatures.items():
                if hasattr(handler_class, method_name):
                    method = getattr(handler_class, method_name)

                    if callable(method) and not isinstance(method, property):
                        actual_sig = inspect.signature(method)

                        ref_params = list(ref_sig.parameters.keys())
                        actual_params = list(actual_sig.parameters.keys())

                        if 'self' in ref_params:
                            ref_params.remove('self')
                        if 'self' in actual_params:
                            actual_params.remove('self')

                        assert ref_params == actual_params, \
                            f"{handler_name}.{method_name} has different parameters. " \
                            f"Expected: {ref_params}, Got: {actual_params}"

                        for param_name in ref_params:
                            ref_param = ref_sig.parameters[param_name]
                            actual_param = actual_sig.parameters[param_name]

                            if ref_param.default != inspect.Parameter.empty:
                                assert actual_param.default == ref_param.default, \
                                    f"{handler_name}.{method_name} parameter '{param_name}' " \
                                    f"has different default. Expected: {ref_param.default}, " \
                                    f"Got: {actual_param.default}"

    def test_step_method_signature_and_return(self, all_handler_classes):
        """Test step method has correct signature and return type."""
        expected_return = tuple[Obs, Reward, Success, TimeOut, Extra]

        for handler_class in all_handler_classes:
            assert hasattr(handler_class, 'step'), \
                f"{handler_class.__name__} missing required method: step"

            method = getattr(handler_class, 'step')
            sig = inspect.signature(method)

            params = list(sig.parameters.keys())
            assert 'self' in params, f"{handler_class.__name__}.step missing 'self' parameter"
            assert 'action' in params, f"{handler_class.__name__}.step missing 'action' parameter"

            if 'action' in sig.parameters:
                param = sig.parameters['action']
                if param.annotation != inspect.Parameter.empty:
                    expected_types = [list[Action], torch.Tensor, "list[Action] | torch.Tensor"]
                    assert any(
                        str(param.annotation).replace(' ', '') == str(expected).replace(' ', '')
                        for expected in expected_types
                    ), f"{handler_class.__name__}.step 'action' parameter has wrong type annotation"

    def test_reset_method_signature_and_return(self, all_handler_classes):
        """Test reset method has correct signature and return type."""
        for handler_class in all_handler_classes:
            assert hasattr(handler_class, 'reset'), \
                f"{handler_class.__name__} missing required method: reset"

            method = getattr(handler_class, 'reset')
            sig = inspect.signature(method)

            params = list(sig.parameters.keys())
            assert 'self' in params, f"{handler_class.__name__}.reset missing 'self' parameter"
            assert 'env_ids' in params, f"{handler_class.__name__}.reset missing 'env_ids' parameter"

            if 'env_ids' in sig.parameters:
                param = sig.parameters['env_ids']
                assert param.default == None, \
                    f"{handler_class.__name__}.reset 'env_ids' should default to None"

    def test_get_states_returns_tensor_state(self, all_handler_classes):
        """Test that _get_states returns TensorState type."""
        for handler_class in all_handler_classes:
            if hasattr(handler_class, '_get_states'):
                method = getattr(handler_class, '_get_states')
                sig = inspect.signature(method)

                if sig.return_annotation != inspect.Parameter.empty:
                    return_str = str(sig.return_annotation)
                    assert 'TensorState' in return_str or 'list[EnvState]' in return_str, \
                        f"{handler_class.__name__}._get_states should return TensorState or list[EnvState]"

    def test_set_pose_method_consistency(self, all_handler_classes):
        """Test set_pose method signature consistency."""
        for handler_class in all_handler_classes:
            if hasattr(handler_class, 'set_pose'):
                method = getattr(handler_class, 'set_pose')
                sig = inspect.signature(method)
                params = list(sig.parameters.keys())

                if 'self' in params:
                    params.remove('self')

                expected_params = ['obj_name', 'pos', 'rot', 'env_ids']
                assert params == expected_params, \
                    f"{handler_class.__name__}.set_pose has wrong parameters. " \
                    f"Expected: {expected_params}, Got: {params}"

                if 'pos' in sig.parameters:
                    param = sig.parameters['pos']
                    if param.annotation != inspect.Parameter.empty:
                        assert 'torch.Tensor' in str(param.annotation), \
                            f"{handler_class.__name__}.set_pose 'pos' should be torch.Tensor"

                if 'rot' in sig.parameters:
                    param = sig.parameters['rot']
                    if param.annotation != inspect.Parameter.empty:
                        assert 'torch.Tensor' in str(param.annotation), \
                            f"{handler_class.__name__}.set_pose 'rot' should be torch.Tensor"

    def test_get_joint_names_behavior(self, all_handler_classes):
        """Test get_joint_names method behavior."""
        for handler_class in all_handler_classes:
            if hasattr(handler_class, 'get_joint_names'):
                method = getattr(handler_class, 'get_joint_names')
                sig = inspect.signature(method)

                params = sig.parameters
                assert 'obj_name' in params, \
                    f"{handler_class.__name__}.get_joint_names missing 'obj_name' parameter"
                assert 'sort' in params, \
                    f"{handler_class.__name__}.get_joint_names missing 'sort' parameter"

                if 'sort' in params:
                    assert params['sort'].default == True, \
                        f"{handler_class.__name__}.get_joint_names 'sort' should default to True"

    def test_device_property_returns_torch_device(self, all_handler_classes):
        """Test that device property returns torch.device."""
        for handler_class in all_handler_classes:
            if hasattr(handler_class, 'device'):
                prop = getattr(handler_class, 'device')
                assert isinstance(prop, property), \
                    f"{handler_class.__name__}.device should be a property"

                if hasattr(prop.fget, '__annotations__'):
                    annotations = prop.fget.__annotations__
                    if 'return' in annotations:
                        assert 'torch.device' in str(annotations['return']), \
                            f"{handler_class.__name__}.device should return torch.device"

    def test_num_envs_property(self, all_handler_classes):
        """Test num_envs property exists and is accessible."""
        for handler_class in all_handler_classes:
            assert hasattr(handler_class, 'num_envs'), \
                f"{handler_class.__name__} missing 'num_envs' property"

            prop = getattr(handler_class, 'num_envs')
            assert isinstance(prop, property), \
                f"{handler_class.__name__}.num_envs should be a property"

    def test_optional_methods_signature_if_present(self, all_handler_classes):
        """Test optional methods have correct signatures if implemented."""
        optional_methods = {
            'get_joint_limits': ['self', 'obj_name', 'joint_name'],
            'set_camera_pose': ['self', 'position', 'look_at'],
            'rand_rigid_body_fric': ['self', 'cfg'],
        }

        for handler_class in all_handler_classes:
            for method_name, expected_params in optional_methods.items():
                if hasattr(handler_class, method_name):
                    method = getattr(handler_class, method_name)
                    if callable(method) and not isinstance(method, property):
                        sig = inspect.signature(method)
                        actual_params = list(sig.parameters.keys())

                        assert actual_params == expected_params, \
                            f"{handler_class.__name__}.{method_name} has wrong parameters. " \
                            f"Expected: {expected_params}, Got: {actual_params}"
