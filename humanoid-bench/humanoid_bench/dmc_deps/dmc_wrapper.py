import mujoco
import numpy as np


class MjDataWrapper:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        return getattr(self._data, name)

    def object_velocity(self, object_id, object_type, local_frame=False):
        """Returns the 6D velocity (linear, angular) of a MuJoCo object.

        Args:
          object_id: Object identifier. Can be either integer ID or String name.
          object_type: The type of the object. Can be either a lowercase string
            (e.g. 'body', 'geom') or an `mjtObj` enum value.
          local_frame: Boolean specifiying whether the velocity is given in the
            global (worldbody), or local (object) frame.

        Returns:
          2x3 array with stacked (linear_velocity, angular_velocity)

        Raises:
          Error: If `object_type` is not a valid MuJoCo object type, or if no object
            with the corresponding name and type was found.
        """
        if not isinstance(object_type, int):
            object_type = _str2type(object_type)
        velocity = np.empty(6, dtype=np.float64)
        if not isinstance(object_id, int):
            object_id = self.model.name2id(object_id, object_type)
        mujoco.mj_objectVelocity(self._model.ptr, self._data, object_type, object_id, velocity, local_frame)
        #  MuJoCo returns velocities in (angular, linear) order, which we flip here.
        return velocity.reshape(2, 3)[::-1]

    def contact_force(self, contact_id):
        """Returns the wrench of a contact as a 2 x 3 array of (forces, torques).

        Args:
          contact_id: Integer, the index of the contact within the contact buffer
            (`self.contact`).

        Returns:
          2x3 array with stacked (force, torque). Note that the order of dimensions
            is (normal, tangent, tangent), in the contact's frame.

        Raises:
          ValueError: If `contact_id` is negative or bigger than ncon-1.
        """
        if not 0 <= contact_id < self.ncon:
            raise ValueError(_CONTACT_ID_OUT_OF_RANGE.format(max_valid=self.ncon - 1, actual=contact_id))

        # Run the portion of `mj_step2` that are needed for correct contact forces.
        mujoco.mj_fwdActuation(self._model.ptr, self._data)
        mujoco.mj_fwdAcceleration(self._model.ptr, self._data)
        mujoco.mj_fwdConstraint(self._model.ptr, self._data)

        wrench = np.empty(6, dtype=np.float64)
        mujoco.mj_contactForce(self._model.ptr, self._data, contact_id, wrench)
        return wrench.reshape(2, 3)

    @property
    def ptr(self):
        """The lower level MjData instance."""
        return self._data

    @property
    def contact(self):
        """Variable-length recarray containing all current contacts."""
        return self._data.contact[: self.ncon]


# Helper functions from dm_control
class MjModelWrapper:
    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    @property
    def ptr(self):
        """The lower level MjModel instance."""
        return self._model

    def __getstate__(self):
        return self._model

    def __setstate__(self, state):
        self._model = state

    def __copy__(self):
        new_model_ptr = copy.copy(self._model)
        return self.__class__(new_model_ptr)

    def name2id(self, name, object_type):
        """Returns the integer ID of a specified MuJoCo object.

        Args:
          name: String specifying the name of the object to query.
          object_type: The type of the object. Can be either a lowercase string
            (e.g. 'body', 'geom') or an `mjtObj` enum value.

        Returns:
          An integer object ID.

        Raises:
          Error: If `object_type` is not a valid MuJoCo object type, or if no object
            with the corresponding name and type was found.
        """
        if isinstance(object_type, str):
            object_type = _str2type(object_type)
        obj_id = mujoco.mj_name2id(self.ptr, object_type, name)
        if obj_id == -1:
            raise Error(f"Object of type {_type2str(object_type)!r} with name {name!r} does not exist.")
        return obj_id

    def id2name(self, object_id, object_type):
        """Returns the name associated with a MuJoCo object ID, if there is one.

        Args:
          object_id: Integer ID.
          object_type: The type of the object. Can be either a lowercase string
            (e.g. 'body', 'geom') or an `mjtObj` enum value.

        Returns:
          A string containing the object name, or an empty string if the object ID
          either doesn't exist or has no name.

        Raises:
          Error: If `object_type` is not a valid MuJoCo object type.
        """
        if isinstance(object_type, str):
            object_type = _str2type(object_type)
        return mujoco.mj_id2name(self.ptr, object_type, object_id) or ""

    @property
    def name(self):
        """Returns the name of the model."""
        # The model's name is the first null terminated string in _model.names
        return str(self._model.names[: self._model.names.find(b"\0")], "ascii")
