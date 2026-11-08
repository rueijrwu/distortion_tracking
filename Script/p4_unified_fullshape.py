"""P4 full relative-field extension with frozen A=0 shape and B4 mean."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p4_sparse_model import L_MM, SPATIAL, SparseModel
from p4_unified_a_gaze_model import FrozenA0, OUTER


def frozen_sparse_base(shape):
    """Convert frozen PF12 gaze powers from /12.4 to SparseModel /20."""
    if shape.order != "PF" or shape.spatial_terms != 12 or len(SPATIAL) != 12:
        raise ValueError("Expected the projective-first frozen PF12 basis")
    if shape.reference_mm.shape != (9, 2) or shape.coefficients.shape != (12,):
        raise ValueError("Frozen P4 reference/coefficients changed")
    factors = np.array([(20. / 12.4)**(ix+iy) for _, ix, iy in SPATIAL])
    return np.asarray(shape.coefficients) * factors


@dataclass
class UnifiedP4Field:
    frozen: FrozenA0
    extension: np.ndarray  # [s1,k1,k2,12 accommodation x spatial interactions]

    def __post_init__(self):
        self.extension = np.asarray(self.extension, float)
        if self.extension.shape != (15,):
            raise ValueError("Expected exactly 15 new P4 field values")
        self.base12 = frozen_sparse_base(self.frozen.p4_shape)

    @property
    def parameter_count(self):
        return 27  # frozen base12 plus new on-axis3 and interaction12

    def sparse_model(self, extension=None):
        values = self.extension if extension is None else np.asarray(extension)
        if values.shape != (15,):
            raise ValueError("Expected 15 extension values")
        return SparseModel(self.frozen.p4_shape.reference_mm, values[:3],
                           np.concatenate((self.base12, values[3:])),
                           1, "projective_first", L_MM)

    def evaluate(self, accommodation_d, gaze_deg, *, extension=None,
                 diagnostics=False, extension_jac=False, state_jac=False):
        gaze = np.asarray(gaze_deg)
        accommodation = np.asarray(accommodation_d)
        if gaze.ndim != 2 or gaze.shape[1] != 2 or accommodation.shape != (len(gaze),):
            raise ValueError("Expected accommodation[N] and gaze[N,ADE,BDE]")
        sparse = self.sparse_model(extension)
        raw = sparse.evaluate(accommodation, gaze[:, 0], gaze[:, 1],
                              diagnostics=diagnostics, theta_jac=extension_jac,
                              calibration_jac=extension_jac, state_jac=state_jac)
        h = raw["prediction"] if isinstance(raw, dict) else raw
        h0_raw = sparse.evaluate(np.zeros(len(gaze)), gaze[:, 0], gaze[:, 1],
                                 state_jac=state_jac)
        h0 = h0_raw["prediction"]
        correction = self.frozen.p4_bias.predict(gaze) - h0.mean(axis=1)
        relative = np.array(h, copy=True)
        relative[:, OUTER] += (9. / 8.) * correction[:, None, :]
        if not diagnostics and not extension_jac and not state_jac:
            return relative
        result = {"prediction": relative,
                  "central_field_zero_error_mm": np.max(np.abs(relative[:, 4]), axis=1),
                  "mean_identity_error_mm": np.max(np.abs(
                      relative.mean(axis=1) -
                      (self.frozen.p4_bias.predict(gaze) + h.mean(axis=1)-h0.mean(axis=1))), axis=1)}
        if diagnostics:
            margins = ("denominator", "radial_tangential", "radial_radial", "brown_determinant")
            result.update({name: raw[name] for name in margins})
            valid = np.isfinite(relative).all(axis=(1, 2))
            for name in margins:
                valid &= np.isfinite(raw[name]).all(axis=1) & np.all(raw[name] > 0, axis=1)
            result["valid"] = valid
        if extension_jac:
            # The fixed mean correction has zero extension derivative: H(0,g)
            # contains only the frozen base, and every extension vanishes at A=0.
            result["extension_jacobian"] = np.concatenate(
                (raw["calibration_jacobian"], raw["theta_jacobian"][..., 12:24]), axis=-1)
        if state_jac:
            derivative = np.array(raw["state_jacobian"], copy=True)
            h0_gaze_derivative = h0_raw["state_jacobian"][..., 1:3].mean(axis=1)
            correction_gaze_derivative = (self.frozen.p4_bias.jacobian(gaze)
                                          - h0_gaze_derivative)
            derivative[:, OUTER, :, 1:3] += (9. / 8.) * correction_gaze_derivative[:, None, :, :]
            result["state_jacobian"] = derivative
        return result


def self_check(frozen):
    model = UnifiedP4Field(frozen, np.zeros(15))
    gaze = np.array([[0., 0.], [3.7, -8.2], [-12.4, 12.4], [12.5, 4.1]])
    at_zero = model.evaluate(np.zeros(len(gaze)), gaze, diagnostics=True)
    reference = frozen.p4_relative_shape(gaze)
    reduction_error = float(np.max(np.abs(at_zero["prediction"]-reference)))
    if reduction_error > 1e-12 or not at_zero["valid"].all():
        raise AssertionError("A=0 full-field reduction differs from frozen PF12+B4")
    values = np.r_[.02, -.01, .003, np.linspace(-.003, .004, 12)]
    accommodation = np.array([0., 1.3, 2.7, 4.])
    analytic = model.evaluate(accommodation, gaze, extension=values, extension_jac=True)["extension_jacobian"]
    numerical = np.empty_like(analytic)
    for column in range(15):
        plus, minus = values.copy(), values.copy()
        plus[column] += 1e-6
        minus[column] -= 1e-6
        numerical[..., column] = ((model.evaluate(accommodation, gaze, extension=plus)-
                                   model.evaluate(accommodation, gaze, extension=minus))/2e-6)
    jac_error = float(np.max(np.abs(analytic-numerical)))
    if jac_error > 1e-7:
        raise AssertionError(f"Restricted 15-value analytic Jacobian differs: {jac_error}")
    state = np.column_stack((accommodation, gaze))
    analytic_state = model.evaluate(accommodation, gaze, extension=values,
                                    state_jac=True)["state_jacobian"]
    numerical_state = np.empty_like(analytic_state)
    for column in range(3):
        plus, minus = state.copy(), state.copy()
        plus[:, column] += 1e-6
        minus[:, column] -= 1e-6
        numerical_state[..., column] = ((model.evaluate(plus[:, 0], plus[:, 1:], extension=values)-
                                         model.evaluate(minus[:, 0], minus[:, 1:], extension=values))/2e-6)
    state_error = float(np.max(np.abs(analytic_state-numerical_state)))
    if state_error > 1e-7:
        raise AssertionError(f"Full F4 state Jacobian differs: {state_error}")
    return {"A0_reduction_max_abs_mm": reduction_error,
            "extension_Jacobian_FD_max_abs_mm": jac_error,
            "state_Jacobian_FD_max_abs_mm": state_error,
            "parameter_count": 27, "new_extension_count": 15}


class ProfileAdapter:
    """The reference401 profiler's positional API for the new F4, not raw H."""

    def __init__(self, field):
        self.field = field

    def evaluate(self, accommodation, ade, bde, *, state_jac=False, diagnostics=False):
        a, y, x = np.broadcast_arrays(np.asarray(accommodation), np.asarray(ade), np.asarray(bde))
        if a.ndim != 1:
            raise ValueError("ProfileAdapter expects one-dimensional aligned state arrays")
        result = self.field.evaluate(a, np.column_stack((y, x)),
                                     state_jac=state_jac, diagnostics=diagnostics)
        return result if isinstance(result, dict) else {"prediction": result}

    def prepare_gaze(self, ade, bde):
        return _PreparedCorrectedGaze(self.field, ade, bde)


class _PreparedCorrectedGaze:
    """Fast profile of F=H+w*Delta0, with a fixed correction per supplied gaze."""

    def __init__(self, field, ade, bde):
        y, x = np.broadcast_arrays(np.asarray(ade), np.asarray(bde))
        if y.ndim != 1:
            raise ValueError("Prepared gaze expects one-dimensional ADE/BDE arrays")
        self.raw = field.sparse_model().prepare_gaze(y, x)
        h0 = self.raw.predict(np.zeros(len(y)))
        self.correction = field.frozen.p4_bias.predict(np.column_stack((y, x)))-h0.mean(axis=1)

    def predict(self, accommodation, indices=None):
        a = np.asarray(accommodation)
        rows = np.arange(len(a)) if indices is None else np.asarray(indices, int)
        if rows.shape != a.shape:
            raise ValueError("Prepared profile row identities must align with accommodation")
        result = np.array(self.raw.predict(a, indices=indices), copy=True)
        result[:, OUTER] += (9. / 8.) * self.correction[rows, None, :]
        return result
