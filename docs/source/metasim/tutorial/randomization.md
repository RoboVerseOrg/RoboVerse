# Randomization

## Design Philosophy

Metasim itself is a deterministic simulation system. In order to mimic the uncertainties we met in real world, we use randomizers to create uncertainties.

There are two types of randomizers: 1) the ones that only changes the simulation result (noise adder), and 2) the ones that modifies the simulation iteself (light, mass, friction, etc.).

The first type of randomizers are easy to implement: simply write a wrapper around the simulation. The second type requires some tricks. We are now adopting the same approach as we constructed extra queries: use hooks to modify the simulation without changing the code of handlers.

Every type 2 randomizers will be called at `reset()` of a simulation.

## Architecture

A custom randomizer can be inherited from `metasim.randomizers.base.BaseRandomizerType`.

This class has only one important method you need to overwrite **__call__()** .

When an Extra Observation is used by a Task, it will be automatically bound to the underlying handler. You can then use `self.handler` to access and modify the handler instance within the Extra Observation class.

When `randomizer()` is called, it needs to do the randomization.

**Currently, the automaticall execution of randomizers in a task pipelien is not implemented.**

## Step-by-Step Usage Guide

TODO
