metasim.cfg.objects
===================

Dependency graph:
.. mermaid::
   %%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%
   flowchart LR
      N0[BaseObjCfg]
      N11[BaseRigidObjCfg]
      N12[BaseArticulationObjCfg]

      subgraph "Level 0"
         N0
      end
      subgraph "Level 1"
         N11
         N12
      end
      subgraph "Level 2"
         RigidObjCfg
         ArticulationObjCfg
         PrimitiveCubeCfg & PrimitiveSphereCfg & PrimitiveCylinderCfg
      end
      subgraph "Mixins"
         direction LR
         FileBasedMixin
         PrimitiveMixin
      end

      N0 --> N11 & N12
      N11 & FileBasedMixin ---> RigidObjCfg
      N12 & FileBasedMixin ---> ArticulationObjCfg
      N11 & PrimitiveMixin ---> PrimitiveCubeCfg & PrimitiveSphereCfg &PrimitiveCylinderCfg


.. automodule:: metasim.cfg.objects

   .. rubric:: Base Class

   .. autosummary::

      BaseObjCfg

   .. rubric:: Primitive Object Classes

   .. autosummary::

      PrimitiveCubeCfg
      PrimitiveSphereCfg
      PrimitiveCylinderCfg

   .. rubric:: File-based Object Classes

   .. autosummary::

      RigidObjCfg
      ArticulationObjCfg

   .. rubric:: Special Object Classes

   .. autosummary::

      NonConvexRigidObjCfg

.. currentmodule:: metasim.cfg.objects

Base Object
-----------

.. autoclass:: BaseObjCfg
   :members:


Primitive Objects
-----------------

.. autoclass:: PrimitiveCubeCfg
   :members:

.. autoclass:: PrimitiveSphereCfg
   :members:

.. autoclass:: PrimitiveCylinderCfg
   :members:

File-based Objects
------------------

.. autoclass:: RigidObjCfg
   :members:

.. autoclass:: ArticulationObjCfg
   :members:

Special Objects
---------------

.. autoclass:: NonConvexRigidObjCfg
   :members:
