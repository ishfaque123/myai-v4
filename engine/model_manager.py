from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    backend: str
    role: str


LOCAL_MODEL = ModelSpec(
    name="qwen3-0.6b-local",
    backend="local",
    role="fast_local_fallback",
)

CLOUD_MODEL = ModelSpec(
    name="gpt-oss-120b-hf",
    backend="cloud",
    role="strong_reasoning_and_agentic_tasks",
)

TOOL_MODEL = ModelSpec(
    name="gpt-oss-120b-hf-tools",
    backend="cloud_tools",
    role="tool_using_tasks",
)

DETERMINISTIC_MODEL = ModelSpec(
    name="nivora-deterministic-tools",
    backend="deterministic",
    role="calculator_or_time",
)


def select_model_plan(route_target, local_available=True, cloud_available=True):
    target = str(route_target or "local")

    if target in ("calculator", "time"):
        return (DETERMINISTIC_MODEL,)

    if target == "web":
        primary = TOOL_MODEL if cloud_available else LOCAL_MODEL
        fallback = LOCAL_MODEL if cloud_available and local_available else None
    elif target == "cloud":
        primary = CLOUD_MODEL if cloud_available else LOCAL_MODEL
        fallback = LOCAL_MODEL if cloud_available and local_available else None
    else:
        primary = LOCAL_MODEL if local_available else CLOUD_MODEL
        fallback = CLOUD_MODEL if local_available and cloud_available else None

    return tuple(model for model in (primary, fallback) if model is not None)


def model_plan_names(route_target, local_available=True, cloud_available=True):
    return [
        model.name
        for model in select_model_plan(
            route_target,
            local_available=local_available,
            cloud_available=cloud_available,
        )
    ]
