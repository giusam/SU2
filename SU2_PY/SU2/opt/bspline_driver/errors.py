"""Driver-specific errors and small parsing wrappers."""

from SU2.opt.bspline_common import normalized_name, parse_float

class BSplineSU2DriverError(RuntimeError):
    pass

class GradientGuardStop(RuntimeError):
    def __init__(self, last_safe_entry, bad_entry, guard_info=None):
        self.last_safe_entry = last_safe_entry
        self.bad_entry = bad_entry
        self.guard_info = guard_info or {}
        super().__init__("Raw-gradient guard stop: rollback to best safe evaluation")

class TrustClipStop(RuntimeError):
    def __init__(self, classification, entry, rollback_entry, diagnostics=None, action=None):
        self.classification = str(classification)
        self.entry = entry
        self.rollback_entry = rollback_entry
        self.diagnostics = diagnostics or {}
        self.action = str(action or "restart_same_level")
        super().__init__(
            f"Trust-clip stop: {self.classification}; action={self.action}"
        )

def _as_float(value, name):
    try:
        return parse_float(value, name, finite=True)
    except ValueError as exc:
        raise BSplineSU2DriverError(str(exc))

def _normalized_name(value):
    return normalized_name(value)
