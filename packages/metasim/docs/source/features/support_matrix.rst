Support Matrix
==============

This page provides an overview of the support for different simulators in MetaSim.


Supported Simulators
--------------------

Backends come in three tiers (``python -m metasim doctor`` prints the installed and tested versions):

- **Tier 1 — verified every release**: ``mujoco`` and ``isaacsim`` (Isaac Sim 6.0 / Isaac Lab 3.0). The benchmark
  suites are replayed on both before a tag; a task that differs between them is a bug.
- **Tier 2 — best effort, run on demand**: ``newton``, ``sapien3``, ``genesis``, ``mjx``, ``superdex``. The handlers are
  maintained and tested when a change touches them; known gaps are listed in the CHANGELOG and ``bench health``
  reports them as ``low`` / ``unsupported`` rather than hiding them.
- **Tier 3 — legacy**: ``isaacgym``, ``pybullet``, ``sapien2``, ``blender``, ``pyrep``. Kept importable, not run in
  CI; fixes are welcome.

The rule across tiers is fail fast, never silently differ: a feature a backend lacks raises or warns once with the
backend named (``unsupported_body_properties``, ``METASIM_SKIP_VERSION_CHECK``), it does not quietly return
something else.


Supported Features
------------------

The following tables show the configuration that can be set for each simulator. Empty cell means the parameter is not supported. ``✓`` means the parameter is supported, and when not specified in the config file, the value read from the asset file or determined by the original simulator is used. Values in the table means the default value to be used when not specified in the config file.


Simulation Configuration
~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 15 15 15

   * - Parameter
     - IsaacSim
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN3
     - PyBullet
     - Newton
   * - ``dt``
     - `1/60 <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.SimulationCfg.dt>`__
     - `1/60 <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=substeps#isaacgym.gymapi.SimParams.substeps>`__
     - `1/500 <https://mujoco.readthedocs.io/en/stable/XMLreference.html#option>`__
     - `1/100 <https://genesis-world.readthedocs.io/en/latest/api_reference/scene/simulator.html#genesis.engine.simulator.Simulator.dt>`__
     - 1/100
     - `1/240 <https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit?tab=t.0#heading=h.kyqqrtg5v8nc>`__
     - 1/60
   * - ``solver_type``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.solver_type>`__
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.PhysXParams.solver_type>`__
     -
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
     - ✓



Robot Configuration
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 15 15 15

   * - Parameter
     - IsaacSim
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN3
     - PyBullet
     - Newton
   * - ``stiffness``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html#isaaclab.actuators.ActuatorBaseCfg.stiffness>`__
     - ✓
     -
     -
     - ✓
     -
     -
   * - ``damping``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html#isaaclab.actuators.ActuatorBaseCfg.damping>`__
     - ✓
     -
     -
     - ✓
     -
     -
   * - ``velocity_limit``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html#isaaclab.actuators.ActuatorBaseCfg.velocity_limit>`__
     -
     -
     -
     -
     -
     -
   * - ``effort_limit_sim``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.actuators.html#isaaclab.actuators.ActuatorBaseCfg.effort_limit>`__
     -
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
     -


Physics Engine Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 15 15 15

   * - Parameter
     - IsaacSim
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN3
     - PyBullet
     - Newton
   * - ``bounce_threshold_velocity``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.bounce_threshold_velocity>`__
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=bounce_threshold_velocity#isaacgym.gymapi.PhysXParams.bounce_threshold_velocity>`__
     -
     -
     -
     -
     -
   * - ``contact_offset``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?#isaacgym.gymapi.RigidShapeProperties.contact_offset>`__
     -
     -
     -
     -
     -
   * - ``friction_correlation_distance``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.friction_correlation_distance>`__
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=friction_correlation_distance#isaacgym.gymapi.PhysXParams.friction_correlation_distance>`__
     -
     -
     -
     -
     -
   * - ``friction_offset_threshold``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.friction_offset_threshold>`__
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=friction_correlation_distance#isaacgym.gymapi.PhysXParams.friction_offset_threshold>`__
     -
     -
     -
     -
     -
   * - ``num_position_iterations``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?#isaacgym.gymapi.PhysXParams.num_position_iterations>`__
     -
     -
     -
     -
     -
   * - ``num_velocity_iterations``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?#isaacgym.gymapi.PhysXParams.num_position_iterations>`__
     -
     -
     -
     -
     -
   * - ``rest_offset``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=rest_offset#isaacgym.gymapi.RigidShapeProperties.rest_offset>`__
     -
     -
     -
     -
     -
   * - ``max_depenetration_velocity``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=max_depenetration_velocity#isaacgym.gymapi.PhysXParams.max_depenetration_velocity:~:text=max_depenetration_velocity>`__
     -
     -
     -
     -
     -
   * - ``default_buffer_size_multiplier``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=max_depenetration_velocity#isaacgym.gymapi.PhysXParams.max_depenetration_velocity:~:text=default_buffer_size_multiplier>`__
     -
     -
     -
     -
     -

Resource Management Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 15 15 15

   * - Parameter
     - IsaacSim
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN3
     - PyBullet
     - Newton
   * - ``num_threads``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.PhysXParams.num_threads>`__
     -
     -
     -
     -
     -

Misc Configuration
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 15 15 15

   * - Parameter
     - IsaacSim
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN3
     - PyBullet
     - Newton
   * - ``replace_cylinder_with_capsule``
     -
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html#isaacgym.gymapi.AssetOptions.replace_cylinder_with_capsule>`__
     -
     -
     -
     -
     -
