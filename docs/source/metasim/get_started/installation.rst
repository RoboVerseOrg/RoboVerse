⚙️ Direct Installation
======================

First, clone the RoboVerse project:

.. code-block:: bash

    git clone git@github.com:RoboVerseOrg/RoboVerse.git && cd RoboVerse

RoboVerse uses `uv <https://docs.astral.sh/uv/>`_ to manage dependencies. To install it, please refer to the `official guide <https://docs.astral.sh/uv/getting-started/installation/>`_, or run:

.. code-block:: bash

    pip install uv

Installation Commands
---------------------

MuJoCo, SAPIEN2, SAPIEN3, Genesis, and PyBullet can be installed directly via ``uv pip install -e ".[<simulator>]"``. However, IsaacLab and IsaacGym must be installed manually.

.. list-table::
   :header-rows: 1
   :widths: 10 30 10 10

   * - Simulator
     - Installation Command
     - Supported Python Versions
     - Recommended Python Version
   * - MuJoCo
     - ``uv pip install -e ".[mujoco]"``
     - 3.9-3.13
     - 3.10-3.11
   * - SAPIEN2
     - ``uv pip install -e ".[sapien2]"``
     - 3.7-3.11
     - 3.10
   * - SAPIEN3
     - ``uv pip install -e ".[sapien3]"``
     - 3.8-3.12
     - 3.10
   * - Genesis
     - ``uv pip install -e ".[genesis]"``
     - 3.10-3.12
     - 3.10-3.11
   * - PyBullet
     - ``uv pip install -e ".[pybullet]"``
     - 3.6-3.11
     - 3.10
   * - IsaacSim v4.5.0
     - See below
     - 3.10
     - 3.10
   * - IsaacSim v5.0.0
     - See below
     - 3.11
     - 3.11
   * - IsaacGym
     - See below
     - 3.6-3.8
     - 3.8

.. note::
   Recommended Python versions are guaranteed to work. For other Python versions, the RoboVerse team hasn't fully tested MetaSim. Please let us know if you encounter any issues.

Please also check the `prerequisites <./prerequisite.html>`_ for supported platforms.

Install IsaacSim v5.0.0 (IsaacLab v2.2.1, Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    uv pip install -e ".[isaacsim]"
    cd third_party
    git clone --depth 1 --branch v2.2.1 git@github.com:isaac-sim/IsaacLab.git IsaacLab221 && cd IsaacLab221
    ./isaaclab.sh -i none

.. note::
   This installation method is only guaranteed to work on Ubuntu 22.04. To install on other platforms, please refer to the `official guide <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>`_.

.. note::
   If you encounter an error such as 'GLIBCXX_3.4.30' not found when running Isaac Sim in a conda environment, try installing GCC 12.1.0 with ``conda install -c conda-forge gcc=12.1.0``. See `Stack Overflow thread <https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin>`_ for more details.

Install Isaacsim v4.5.0 (IsaacLab v2.1.1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    uv pip install -e ".[isaacsim211]"
    cd third_party
    git clone --depth 1 --branch v2.1.1 git@github.com:isaac-sim/IsaacLab.git IsaacLab211 && cd IsaacLab211
    ./isaaclab.sh -i none

.. note::
   This installation method is only guaranteed to work on Ubuntu 22.04. To install on other platforms, please refer to the `official guide <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html>`_.


Install IsaacGym
~~~~~~~~~~~~~~~~

.. code-block:: bash

    cd third_party
    wget https://developer.nvidia.com/isaac-gym-preview-4 \
        && tar -xf isaac-gym-preview-4 \
        && rm isaac-gym-preview-4
    find isaacgym/python -type f -name "*.py" -exec sed -i 's/np\.float/np.float32/g' {} +
    cd ..
    uv pip install -e ".[isaacgym]" 'isaacgym @ file:${PROJECT_ROOT}/third_party/isaacgym/python'

.. note::
   This installation method is only guaranteed to work on Ubuntu 22.04. To install on other platforms, you can refer to the `clone <https://docs.robotsfan.com/isaacgym/install.html>`_ of the official guide.

.. tip::
   If you encounter the error ``FileNotFoundError: [Errno 2] No such file or directory: '.../lib/python3.8/site-packages/isaacgym/_bindings/src/gymtorch/gymtorch.cpp'``, you can try to run the following command:

   .. code-block:: bash

      mkdir -p $CONDA_PREFIX/lib/python3.8/site-packages/isaacgym/_bindings/src
      cp -r third_party/isaacgym/python/isaacgym/_bindings/src/gymtorch $CONDA_PREFIX/lib/python3.8/site-packages/isaacgym/_bindings/src/gymtorch

   If you encounter the error ``ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory``, you can try to run the following command:

   .. code-block:: bash

      export LD_LIBRARY_PATH=$CONDA_HOME/envs/metasim_isaacgym/lib:$LD_LIBRARY_PATH

   where ``$CONDA_HOME`` is the path to your conda installation. It is typically ``~/anaconda3``, ``~/miniconda3`` or ``~/miniforge3``.
   You can also add it to your ``~/.bashrc`` to make it permanent.

Install Multiple Simulators
---------------------------

Feel free to combine the above commands to install multiple simulators in one environment. For example, to install MuJoCo and IsaacSim at the same time, you can run:

.. code-block:: bash

    uv pip install -e ".[mujoco,isaaclab221]"
    cd third_party
    git clone --depth 1 --branch v2.2.1 git@github.com:isaac-sim/IsaacLab.git IsaacLab221 && cd IsaacLab221
    ./isaaclab.sh -i none

.. note::
   Every time you install multiple simulators, you need to use one single command to deal with dependencies correctly. For example, to install MuJoCo, SAPIEN3, and Genesis at the same time, you should run:

   .. code-block:: bash

      uv pip install -e ".[mujoco,sapien3,genesis]"

   instead of running them one by one:

   .. code-block:: bash

      uv pip install -e ".[mujoco]"
      uv pip install -e ".[sapien3]"
      uv pip install -e ".[genesis]"
