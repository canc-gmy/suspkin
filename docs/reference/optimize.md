# Optimize (experimental)

!!! warning "Experimental"
    `suspkin.optimize` may change in minor releases. The study YAML schema 1 is stable; see
    [Stability and versioning](../guides/stability.md) and the [Study YAML](../guides/study-yaml.md)
    guide.

## Study specification

::: suspkin.optimize.spec
    options:
      members:
        - load_study
        - Study
        - StudyConfigError

## Design space

::: suspkin.optimize.design
    options:
      members:
        - DesignSpace
        - InfeasibleDesign

## Evaluation

::: suspkin.optimize.evaluate
    options:
      members:
        - evaluate
        - Evaluation
        - ConstraintResult
        - ObjectiveResult

## Problem and history

::: suspkin.optimize.problem
    options:
      members:
        - Problem
        - BudgetExhausted

::: suspkin.optimize.history
    options:
      members:
        - HistoryWriter
        - EvalCache
        - read_history

## Drivers

::: suspkin.optimize.drivers
    options:
      members:
        - run_optimization
        - RunResult
