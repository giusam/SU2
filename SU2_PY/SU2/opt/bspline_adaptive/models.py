"""Data models for progressive B-spline adaptation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from SU2.opt.bspline_driver.reduction import active_mode_ids

@dataclass
class BsplineLevel:
    level_id: int
    workdir: Path
    opt_workdir: Path
    active_modes: dict
    active_modes_start_filename: Path
    optimized_modes_filename: Path
    selection_metadata: dict
    initial_modes_source: Path

    @property
    def ndv(self):
        return len(self.active_mode_ids)

    @property
    def active_mode_ids(self):
        return active_mode_ids(self.active_modes)

@dataclass
class TriggerDecision:
    trigger_mode: str
    metric: object = ""
    threshold: object = ""
    window: object = ""
    patience: object = ""
    counter: int = 0
    refine_now: bool = False
    reason: str = ""

@dataclass
class ClampedSideGroup:
    side: str
    degree: int
    knot_vector: tuple
    modes: list

    @property
    def n_basis(self):
        return len(self.modes)

    @property
    def coefficients(self):
        return np.asarray(
            [float(mode.get("coefficient", 0.0)) for mode in self.modes],
            dtype=float,
        )

@dataclass
class ClampedKnotSpace:
    spec: dict
    groups: dict
    sides: tuple
    degree: int
    knot_vector: tuple
    coupling: str

    @property
    def modes(self):
        modes = []
        for side in self.sides:
            modes.extend(self.groups[side].modes)
        return modes
