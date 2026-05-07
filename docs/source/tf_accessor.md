# Transfer-Function Dataset Accessor (`ds.tf`)

## Overview

MTpy-v2 stores every MT transfer function as an `xarray.Dataset` whose data
variables (`transfer_function`, `transfer_function_error`,
`transfer_function_model_error`) use labelled `output`, `input`, and `period`
dimensions.  The **`tf` accessor** is registered on every such dataset so that
all derived quantities — impedance, phase tensor, tipper, and correction
routines — are reachable through a single, discoverable entry point:

```python
ds.tf.<property_or_method>
```

The accessor is the single source of truth for transfer-function math.  Both
the `MT` object and the `MTData` collection delegate their core operations
(rotate, interpolate, remove_static_shift, remove_distortion) to it.

---

## Architecture

```
xr.Dataset  (period × output × input)
│
└── .tf   ←  TFDatasetAccessor  (registered via @xr.register_dataset_accessor)
      │
      ├── Data access
      │     z(), z_error(), z_model_error()
      │     tipper(), tipper_error(), tipper_model_error()
      │
      ├── Derived Z quantities  (properties, all lazy)
      │     resistivity, phase
      │     resistivity_error, phase_error
      │     resistivity_model_error, phase_model_error
      │     Per-component aliases: res_xx / res_xy / res_yx / res_yy
      │                            phase_xx / phase_xy / phase_yx / phase_yy
      │     Determinant variants: det, res_det, phase_det  (+ error counterparts)
      │
      ├── Tipper quantities
      │     t_zx, t_zy  (+ error variants)
      │     tipper_amplitude, tipper_phase, tipper_mag_real/imag, tipper_angle_real/imag
      │
      ├── Phase tensor quantities
      │     pt_xx / pt_xy / pt_yx / pt_yy  (+ error counterparts)
      │     pt_phimax / pt_phimin, pt_azimuth, pt_skew
      │     pt_ellipticity, pt_eccentricity, pt_trace, pt_det
      │     pt_alpha, pt_beta  (+ all error counterparts)
      │     phase_tensor  →  PhaseTensor object
      │     invariants    →  dict of standard PT invariants
      │
      ├── Object converters
      │     to_z()       →  Z
      │     to_tipper()  →  Tipper
      │     to_pt()      →  PhaseTensor
      │
      ├── Builders  (write back into the dataset)
      │     with_z(z_obj)         →  updates transfer_function* variables
      │     with_tipper(t_obj)    →  updates tipper channel block
      │     with_res_phase(...)   →  constructs Z from ρ/φ, then writes
      │     set_resistivity_phase(...)  →  alias
      │     set_amp_phase(...)          →  Tipper setter alias
      │     set_mag_direction(...)      →  Tipper setter alias
      │
      ├── Operations  (return new Dataset or modify in-place)
      │     rotate(alpha, coordinate_reference_frame='ned', inplace=False)
      │     interpolate(new_periods, method='slinear', inplace=False)
      │
      └── Correction / analysis helpers
            remove_ss(ss_x, ss_y, as_dataset=False)
            remove_distortion(distortion_tensor, as_dataset=False)
            estimate_dimensionality()
            estimate_distortion()
            estimate_depth_of_investigation()
```

### Where the math lives

All numerical formulas are collected in
`mtpy/core/transfer_function/tf_helpers.py`.  The legacy `Z`, `Tipper`, and
`PhaseTensor` classes import from the same module, so there is exactly one
implementation of every formula shared across the entire stack.

---

## Integration with `MT` and `MTData`

### `MT`

The `MT` object stores its transfer function as `self._transfer_function` (an
`xr.Dataset`).  The accessor is therefore always available as:

```python
mt._transfer_function.tf
```

The following `MT` methods delegate entirely to the accessor:

| `MT` method | Accessor call |
|---|---|
| `mt.rotate(angle)` | `self._transfer_function.tf.rotate(angle, inplace=True)` |
| `mt.interpolate(periods)` | `self._transfer_function.tf.interpolate(periods)` |
| `mt.remove_static_shift(sx, sy)` | `self._transfer_function.tf.remove_ss(sx, sy, as_dataset=True)` |
| `mt.remove_distortion(D)` | `self._transfer_function.tf.remove_distortion(D, as_dataset=True)` |

### `MTData`

`MTData` is a tree-backed collection of `MT` stations stored as `xr.Dataset`
nodes inside an `xr.DataTree`.  The two per-station static helpers that were
previously large self-contained routines now delegate to the accessor:

| `MTData` helper | Accessor call |
|---|---|
| `_rotate_station_dataset(ds, angle, crf)` | `ds.load().tf.rotate(angle, coordinate_reference_frame=crf)` |
| `_interpolate_station_dataset(ds, periods)` | `ds.tf.interpolate(periods)` |

Bulk methods `rotate()`, `interpolate()`, `rotate_dask()`, `interpolate_dask()`,
and `interpolate_lazy()` all flow through these helpers and therefore implicitly
use the accessor.

---

## Usage Examples

### Accessing derived quantities

```python
from mtpy import MT

mt = MT("path/to/station.edi")
mt.read()

ds = mt._transfer_function  # xr.Dataset

# Impedance tensor (complex, shape n_periods × 2 × 2)
z_array = ds.tf.z()

# Apparent resistivity and phase (shape n_periods × 2 × 2)
rho = ds.tf.resistivity
phi = ds.tf.phase

# Off-diagonal components only
rho_xy = ds.tf.res_xy   # shape (n_periods,)
phi_yx = ds.tf.phase_yx

# Determinant resistivity
rho_det = ds.tf.res_det

# Phase tensor azimuth
azimuth = ds.tf.pt_azimuth
```

### Converting to legacy objects

```python
z_obj     = ds.tf.to_z()       # Z instance
t_obj     = ds.tf.to_tipper()  # Tipper instance
pt_obj    = ds.tf.to_pt()      # PhaseTensor instance
```

### Rotating a dataset

```python
# Returns a new dataset rotated 30 degrees clockwise (NED convention)
rotated_ds = ds.tf.rotate(30)

# In-place rotation
ds.tf.rotate(30, inplace=True)
```

### Interpolating to a new period grid

```python
import numpy as np

new_periods = np.logspace(-3, 3, 50)
interp_ds = ds.tf.interpolate(new_periods)

# The resulting dataset has len(new_periods) entries along the period axis
print(interp_ds.tf.z().shape)   # (50, 2, 2)
```

### Removing static shift and distortion

```python
# Remove static shift: sx applied to Ex row, sy to Ey row
corrected_ds = ds.tf.remove_ss(sx=0.8, sy=1.2, as_dataset=True)

# Remove a known distortion tensor
import numpy as np
D = np.array([[1.1, 0.05], [0.02, 0.95]])
undistorted_ds = ds.tf.remove_distortion(D, as_dataset=True)
```

### Updating impedance values (builder pattern)

```python
from mtpy.core.transfer_function.z import Z

# Create or modify a Z object then write it back
z_obj = ds.tf.to_z()
z_obj.z[:, 0, 0] = 0.0  # zero out Zxx
ds.tf.with_z(z_obj, inplace=True)
```

### Builder methods (`with_*`) end-to-end

```python
import numpy as np

# Start from an existing station dataset
ds = mt._transfer_function

# 1) Build/update impedance directly with with_z
z_obj = ds.tf.to_z()
z_obj.z[:, 0, 0] = np.nan   # clear Zxx
z_obj.z[:, 1, 1] = np.nan   # clear Zyy
ds1 = ds.tf.with_z(z_obj, inplace=False)

# 2) Build/update tipper directly with with_tipper
t_obj = ds1.tf.to_tipper()
t_obj.tipper[:, 0, 0] *= 0.95
t_obj.tipper[:, 0, 1] *= 1.05
ds2 = ds1.tf.with_tipper(t_obj, inplace=False)

# 3) Construct impedance from resistivity/phase with with_res_phase
rho = ds2.tf.resistivity
phi = ds2.tf.phase

# Example transformation: increase off-diagonal rho by 10%
rho_mod = rho.copy()
rho_mod[:, 0, 1] *= 1.10
rho_mod[:, 1, 0] *= 1.10

ds3 = ds2.tf.with_res_phase(
      resistivity=rho_mod,
      phase=phi,
      period=ds2.period.values,
      inplace=False,
)

# Optional aliases (equivalent builder entry points)
ds4 = ds3.tf.set_resistivity_phase(
      resistivity=ds3.tf.resistivity,
      phase=ds3.tf.phase,
      period=ds3.period.values,
      inplace=False,
)
```

### Tipper builder aliases

```python
# Set tipper from amplitude/phase
amp = ds.tf.tipper_amplitude
phs = ds.tf.tipper_phase
ds_amp = ds.tf.set_amp_phase(amp, phs, inplace=False)

# Set tipper from magnitude/direction pairs
mag_real = ds_amp.tf.tipper_mag_real
ang_real = ds_amp.tf.tipper_angle_real
mag_imag = ds_amp.tf.tipper_mag_imag
ang_imag = ds_amp.tf.tipper_angle_imag
ds_mag = ds_amp.tf.set_mag_direction(
      mag_real,
      ang_real,
      mag_imag,
      ang_imag,
      inplace=False,
)
```

### Working via `MT`

The `MT` API is the recommended entry point for single-station workflows.
All the above operations are available directly:

```python
from mtpy import MT

mt = MT("station.edi")
mt.read()

mt.rotate(30)                              # rotates in place
mt.interpolate(np.logspace(-2, 2, 40))    # returns a new MT
mt.remove_static_shift(ss_x=0.8, ss_y=1.2)
mt.remove_distortion()
```

### Working via `MTData`

For multi-station workflows use `MTData`, which applies the same accessor
operations to every station in the collection:

```python
from mtpy.core import MTData

md = MTData()
md.add_stations(mt_list)

# Rotate all stations by 15 degrees
md.rotate(15)

# Interpolate all stations to a shared period grid
periods = np.logspace(-3, 3, 60)
md.interpolate(periods)

# Access the underlying xarray dataset for one station
station_ds = md.get_station("surveyA/st01")
print(station_ds.tf.res_xy)
```

---

## Testing

The accessor is covered by
`tests/core/transfer_function/test_tf_accessor.py` (26+ test methods) and
`tests/core/test_mt.py::TestMTAccessorIntegration` (14 tests verifying that
`MT` methods correctly round-trip through the accessor).
