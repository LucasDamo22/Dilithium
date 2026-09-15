"""Alternative 2D visualization modes sharing the session's trace."""

from __future__ import annotations

from typing import List, Tuple

from PyQt5 import QtWidgets

from ..session import Session


def make_modes(session: Session) -> List[Tuple[str, QtWidgets.QWidget]]:
    from .slices import SliceStack
    from .lanes import LaneTable
    from .bytesview import BytesView
    from .heatmap import DiffusionHeatmap
    from .avalanche import AvalancheView
    from .interleaved import InterleavedView
    from .batched import BatchedView
    from .sponge_view import SpongeView

    return [
        ("slices", SliceStack(session)),
        ("lanes", LaneTable(session)),
        ("bytes", BytesView(session)),
        ("heatmap", DiffusionHeatmap(session)),
        ("avalanche", AvalancheView(session)),
        ("interleaved", InterleavedView(session)),
        ("batched", BatchedView(session)),
        ("sponge", SpongeView(session)),
    ]
