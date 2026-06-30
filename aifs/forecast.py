import os
import datetime
from typing import Generator

# Apply flash-attn shim before any anemoi import
import aifs.compat  # noqa: F401
from aifs.device import get_device, device_label

DEFAULT_CHECKPOINT = "aifs-single-2.0"

CHECKPOINTS = {
    DEFAULT_CHECKPOINT: {"huggingface": f"ecmwf/{DEFAULT_CHECKPOINT}"},
    "aifs-ens-2.0": {"huggingface": f"ecmwf/aifs-ens-2.0"},
    "aifs-single-1.1": {"huggingface": f"ecmwf/aifs-single-1.1"},
    "aifs-ens-1.0": {"huggingface": f"ecmwf/aifs-ens-1.0"},
}


def run_forecast(
    fields: dict,
    date: datetime.datetime,
    lead_time: int = 24,
    num_chunks: int = 16,
    checkpoint: str = DEFAULT_CHECKPOINT,
    verbose: bool = True,
):
    """
    Same as before, but now a generator.

    Yields
    ------
    str  -- progress messages
    Final value (via StopIteration / return) is the list of state dicts.
    Use `yield from` pattern below to capture it on the caller side.
    """
    if lead_time % 6 != 0:
        raise ValueError(f"lead_time must be a multiple of 6, got {lead_time}")

    from anemoi.inference.runners.simple import SimpleRunner

    device = get_device()

    if device == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(num_chunks)

    if verbose:
        yield "log", f"🖥️   Device  : {device_label()} \n\n Checkpoint: {checkpoint} \n\n Lead time : {lead_time} h  ({lead_time // 6} steps)"

    ckpt = CHECKPOINTS.get(checkpoint, checkpoint)

    if verbose:
        yield "log" , "🤖  Loading model …"

    runner = SimpleRunner(ckpt)

    if verbose:
        yield "log" , "🌍  Running inference …"

    states: list[dict] = []
    input_state = {"fields": fields, "date": date}
    for state in runner.run(input_states=input_state, lead_time=lead_time):
        states.append({
            "date": state["date"],
            "fields": {k: v.copy() for k, v in state["fields"].items()},
            "latitudes": state["latitudes"],
            "longitudes": state["longitudes"]
        })
        if verbose:
            yield "log", f"    ✓  {state['date']}"

        if verbose:
            yield "log", f"✅  Done — {len(states)} steps produced."

        yield "result", states


def run_forecast_streaming(
    fields: dict,
    date: datetime.datetime,
    lead_time: int = 24,
    num_chunks: int = 16,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> Generator[dict, None, None]:
    """
    Generator variant of :func:`run_forecast`.

    Yields each state dict as soon as it is computed, which is useful for
    Gradio apps or notebooks that want to display results incrementally.

    Example
    -------
        for state in run_forecast_streaming(fields, date, lead_time=48):
            plot_field(state)
    """
    if lead_time % 6 != 0:
        raise ValueError(f"lead_time must be a multiple of 6, got {lead_time}")

    from anemoi.inference.runners.simple import SimpleRunner

    device = get_device()
    if device == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["ANEMOI_INFERENCE_NUM_CHUNKS"] = str(num_chunks)

    ckpt = CHECKPOINTS.get(checkpoint, checkpoint)
    runner = SimpleRunner(ckpt)

    input_state = {"fields": fields, "date": date}
    for state in runner.run(input_states=input_state, lead_time=lead_time):
        yield {
            "date": state["date"],
            "fields": {k: v.copy() for k, v in state["fields"].items()}
    }
