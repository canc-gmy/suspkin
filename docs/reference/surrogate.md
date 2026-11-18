# Surrogate (experimental)

!!! warning "Experimental"
    `suspkin.surrogate` may change in minor releases and is not exported in `suspkin.__all__`.
    Training and inference need the `surrogate` extra (`uv sync --extra surrogate`). The study YAML
    `surrogate` block is part of the study schema 1 and is documented in the
    [Study YAML](../guides/study-yaml.md) guide. The design and limitations are in
    [Architecture section 15](../ARCHITECTURE.md#15-surrogate-v11-experimental).

## Channels and dataset

::: suspkin.surrogate.channels
    options:
      members:
        - Channel
        - CHANNELS
        - channels_for

::: suspkin.surrogate.dataset
    options:
      members:
        - sample_axle
        - load_dataset
        - DatasetError

## Training and validation

::: suspkin.surrogate.train
    options:
      members:
        - TrainConfig
        - TrainedModel
        - resolve_device

::: suspkin.surrogate.workflow
    options:
      members:
        - train_surrogate
        - validate_surrogate

## Vehicle layer

::: suspkin.surrogate.vehicle_layer
    options:
      members:
        - vehicle_layer
        - layer_error
        - LayerResult

## Optimization on the networks

::: suspkin.surrogate.evaluate
    options:
      members:
        - SurrogateEvaluator

::: suspkin.surrogate.problem
    options:
      members:
        - SurrogateProblem

::: suspkin.surrogate.active
    options:
      members:
        - run_active_learning
        - ActiveResult
