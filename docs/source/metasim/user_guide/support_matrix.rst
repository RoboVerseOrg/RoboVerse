Support Matrix
==============

This page provides a matrix of the support for different simulators in MetaSim.

Physics Engine Parameters
-------------------------

The following table shows the parameters that can be set for the physics engine in each simulator.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20 20 20

   * - Parameter
     - IsaacLab v2.0.2
     - IsaacGym
     - MuJoCo
     - Genesis
     - SAPIEN v3
     - PyBullet
   * - ``friction_offset_threshold``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.friction_offset_threshold>`_
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=friction_correlation_distance#isaacgym.gymapi.PhysXParams.friction_offset_threshold>`_
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
   * - ``bounce_threshold_velocity``
     - `✓ <https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.bounce_threshold_velocity>`_
     - `✓ <https://docs.robotsfan.com/isaacgym/api/python/struct_py.html?highlight=bounce_threshold_velocity#isaacgym.gymapi.PhysXParams.bounce_threshold_velocity>`_
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
