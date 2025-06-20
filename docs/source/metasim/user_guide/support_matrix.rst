Support Matrix
==============

This page provides an overview of the support for different simulators in MetaSim.


Supported Simulators
--------------------

There are 3 levels of supportance for each simulator:

- **Actively supported**: ``isaaclab``, ``isaacgym``, ``mujoco``, ``sapien3``, ``genesis``. These simulators should always be guaranteed to work on the main branch.
- **Inactively supported**: ``pybullet``, ``pyrep``. These simulators won't be actively supported. They will only be guaranteed to work when a major version is released. Note that ``sapien2`` will be deprecated when ``sapien3`` exits `beta <https://github.com/haosulab/SAPIEN/releases>`_.
- **Experimental**: ``mjx``, ``blender``. These simulators (renderers) are still in experimental stage and will be added to "actively supported" list in the future.


Supported Features
------------------

The following tables show the configuration that can be set for each simulator. Empty cell means the parameter is not supported. ``✓`` means the parameter is supported, and when not specified in the config file, the value read from the asset file or determined by the original simulator is used. Values in the table means the default value to be used when not specified in the config file.


Simulation Configuration
~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20 20 20

   * - Parameter
     - IsaacLab
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN3
     - PyBullet
   * - ``dt``
     - `1/60 <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.SimulationCfg.dt>`_
     - `1/60 <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=substeps#isaacgym.gymapi.SimParams.substeps>`_
     - `1/500 <https://mujoco.readthedocs.io/en/stable/XMLreference.html#option>`_
     - `1/100 <https://genesis-world.readthedocs.io/en/latest/api_reference/scene/simulator.html#genesis.engine.simulator.Simulator.dt>`_
     - 1/100
     - `1/240 <https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit?tab=t.0#heading=h.kyqqrtg5v8nc>`_
   * - ``solver_type``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.solver_type>`_
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.PhysXParams.solver_type>`_
     -
     -
     -
     -
   * - ``env_spacing``
     - 
     - ✓
     -
     - ✓
     -
     -



Robot Configuration
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20 20 20

   * - Parameter
     - IsaacLab
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN3
     - PyBullet
   * - ``stiffness``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html#isaaclab.actuators.ActuatorBaseCfg.stiffness>`_
     - ✓
     -
     -
     - ✓
     -
   * - ``damping``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html#isaaclab.actuators.ActuatorBaseCfg.damping>`_
     - ✓
     -
     -
     - ✓
     -
   * - ``velocity_limit``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html#isaaclab.actuators.ActuatorBaseCfg.velocity_limit>`_
     -
     -
     -
     -
     -
   * - ``torque_limit``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html#isaaclab.actuators.ActuatorBaseCfg.effort_limit>`_
     -
     -
     -
     -
     -
   * - ``fully_actuated``
     - ✓
     - ✓
     -
     -
     -
     -


Physics Engine Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20 20 20

   * - Parameter
     - IsaacLab
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN3
     - PyBullet
   * - ``bounce_threshold_velocity``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.bounce_threshold_velocity>`_
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=bounce_threshold_velocity#isaacgym.gymapi.PhysXParams.bounce_threshold_velocity>`_
     -
     -
     -
     -
   * - ``contact_offset``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?#isaacgym.gymapi.RigidShapeProperties.contact_offset>`_
     -
     -
     -
     -
   * - ``friction_correlation_distance``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.friction_correlation_distance>`_
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=friction_correlation_distance#isaacgym.gymapi.PhysXParams.friction_correlation_distance>`_
     -
     -
     -
     -
   * - ``friction_offset_threshold``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.friction_offset_threshold>`_
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=friction_correlation_distance#isaacgym.gymapi.PhysXParams.friction_offset_threshold>`_
     -
     -
     -
     -
   * - ``num_position_iterations``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?#isaacgym.gymapi.PhysXParams.num_position_iterations>`_
     -
     -
     -
     -
   * - ``num_velocity_iterations``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?#isaacgym.gymapi.PhysXParams.num_position_iterations>`_
     -
     -
     -
     -
   * - ``rest_offset``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=rest_offset#isaacgym.gymapi.RigidShapeProperties.rest_offset>`_
     -
     -
     -
     -
   * - ``max_depenetration_velocity``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=max_depenetration_velocity#isaacgym.gymapi.PhysXParams.max_depenetration_velocity:~:text=max_depenetration_velocity>`_
     -
     -
     -
     -
   * - ``default_buffer_size_multiplier``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=max_depenetration_velocity#isaacgym.gymapi.PhysXParams.max_depenetration_velocity:~:text=default_buffer_size_multiplier>`_
     -
     -
     -
     -

Resource Management Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20 20 20

   * - Parameter
     - IsaacLab
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN3
     - PyBullet
   * - ``num_threads``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.PhysXParams.num_threads>`_
     -
     -
     -
     -

Misc Configuration
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20 20 20

   * - Parameter
     - IsaacLab
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN3
     - PyBullet
   * - ``replace_cylinder_with_capsule``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.AssetOptions.replace_cylinder_with_capsule>`_
     -
     -
     -
     -
