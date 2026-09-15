"""JSON export of traces and sponge runs."""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np

from . import keccak as K
from . import sponge as S


def _lanes_hex(state: np.ndarray):
    """[[hex lane for y in 0..4] for x in 0..4] - indexed [x][y] like the spec."""
    return [[f"{int(state[x, y]):016x}" for y in range(5)] for x in range(5)]


def snapshot_to_dict(snap: K.Snapshot) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "index": snap.index,
        "round": snap.round,
        "step": snap.step,
        "skipped": snap.skipped,
        "lanes_hex_xy": _lanes_hex(snap.state),
        "state_bytes_hex": K.state_to_bytes(snap.state).hex(),
    }
    if snap.theta_c is not None:
        d["theta_C"] = [f"{int(v):016x}" for v in snap.theta_c]
        d["theta_D"] = [f"{int(v):016x}" for v in snap.theta_d]
    if snap.round_constant is not None:
        d["round_constant"] = f"{snap.round_constant:016x}"
        d["round_index_abs"] = snap.round_index_abs
    return d


def trace_to_dict(trace: K.Trace) -> Dict[str, Any]:
    return {
        "num_rounds": trace.num_rounds,
        "round_offset": trace.round_offset,
        "enabled_steps": list(trace.enabled_steps),
        "rho_offsets_xy": K.RHO_OFFSETS.tolist(),
        "round_constants": [f"{c:016x}" for c in K.ROUND_CONSTANTS],
        "snapshots": [snapshot_to_dict(s) for s in trace.snapshots],
    }


def run_to_dict(run: S.SpongeRun) -> Dict[str, Any]:
    v = run.variant
    return {
        "variant": {
            "name": v.name,
            "rate_bytes": v.rate_bytes,
            "capacity_bytes": v.capacity_bytes,
            "domain_byte": f"{v.domain_byte:02x}",
            "output_bytes": run.output_bytes,
        },
        "num_rounds": run.num_rounds,
        "enabled_steps": list(run.enabled_steps),
        "message_hex": run.message.hex(),
        "padded_hex": run.padded.hex(),
        "output_hex": run.output.hex(),
        "absorb_blocks": [
            {"index": b.index, "data_hex": b.data.hex(), "is_last": b.is_last,
             "perm_call": b.perm.index}
            for b in run.absorb_blocks
        ],
        "squeeze_blocks": [
            {"index": s.index, "data_hex": s.data.hex(),
             "perm_call": None if s.perm is None else s.perm.index}
            for s in run.squeeze_blocks
        ],
        "perm_calls": [
            {"index": c.index, "phase": c.phase, "block_index": c.block_index,
             "state_in_hex": K.state_to_bytes(c.state_in).hex(),
             "state_out_hex": K.state_to_bytes(c.state_out).hex(),
             "trace": None if c.trace is None else trace_to_dict(c.trace)}
            for c in run.perm_calls
        ],
    }


def run_to_json(run: S.SpongeRun, indent: int = 1) -> str:
    return json.dumps(run_to_dict(run), indent=indent)


def trace_to_json(trace: K.Trace, indent: int = 1) -> str:
    return json.dumps(trace_to_dict(trace), indent=indent)


__all__ = ["snapshot_to_dict", "trace_to_dict", "run_to_dict", "run_to_json", "trace_to_json"]
