metasim.cfg.checkers
====================

.. automodule:: metasim.cfg.checkers

   .. rubric:: Base Class

   .. currentmodule:: metasim.cfg.checkers._base_checker

   .. autosummary::

      BaseChecker

   .. rubric:: Checker Operators

   .. currentmodule:: metasim.cfg.checkers._checker_operators

   .. autosummary::

      AndOp
      OrOp
      NotOp

   .. rubric:: Checkers

   .. currentmodule:: metasim.cfg.checkers._checkers

   .. autosummary::

      DetectedChecker
      JointPosChecker
      PositionShiftChecker
      RotationShiftChecker
      JointPosShiftChecker

   .. rubric:: Detectors

   .. currentmodule:: metasim.cfg.checkers._detectors

   .. autosummary::

      RelativeBboxDetector
      Relative3DSphereDetector

.. currentmodule:: metasim.cfg.checkers


Base Checker
------------

.. autoclass:: BaseChecker
   :members:

Checker Operators
-----------------

.. autoclass:: AndOp
   :members:

.. autoclass:: OrOp
   :members:

.. autoclass:: NotOp
   :members:

Checkers
--------

.. autoclass:: DetectedChecker
   :members:
   :show-inheritance:

.. autoclass:: JointPosChecker
   :members:
   :show-inheritance:

.. autoclass:: PositionShiftChecker
   :members:
   :show-inheritance:

.. autoclass:: RotationShiftChecker
   :members:
   :show-inheritance:

.. autoclass:: JointPosShiftChecker
   :members:
   :show-inheritance:

Detectors
---------

.. autoclass:: RelativeBboxDetector
   :members:
   :show-inheritance:

.. autoclass:: Relative3DSphereDetector
   :members:
   :show-inheritance:
