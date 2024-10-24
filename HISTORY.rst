History
=======

2.0.0 (2022-10-20)
---------------------

* major changes under the hood.  
    - Now using `mt_metadata` to read/write transfer function files
	- Now using `mth5` to store the transfer functions
	- Introduced `MT`, `MTCollection`, and `MTData` such that operations are more centralized. Now most methods can be called from `MT` and `MTData`
	- Removing older modules and group specific modules
	- Added GitActions for testing
	- Updating tests (still lots of work to do)
	- Updated documentation to upload to ReadTheDocs

2.0.5 (2023-11-09)
---------------------

* bug fixes
* now install simpeg for inversions, 1D implemented so far

2.0.8 (2024-08-30)
---------------------

* Shapefiles by @kujaku11 in https://github.com/MTgeophysics/mtpy-v2/pull/28
* Added to ModEM module docs by @oaazeved in https://github.com/MTgeophysics/mtpy-v2/pull/32
* Add tf quality by @kujaku11 in https://github.com/MTgeophysics/mtpy-v2/pull/37
* Updates by @kujaku11 in https://github.com/MTgeophysics/mtpy-v2/pull/29
* adding ability to just plot tipper by @kujaku11 in https://github.com/MTgeophysics/mtpy-v2/pull/35
* Updated default tippers to point towards a good conductor by @kujaku11 in https://github.com/MTgeophysics/mtpy-v2/pull/41
* Ak2 by @alkirkby in https://github.com/MTgeophysics/mtpy-v2/pull/43
* Porting Aurora objects by @kujaku11 in https://github.com/MTgeophysics/mtpy-v2/pull/45
* Updates by @kujaku11 in https://github.com/MTgeophysics/mtpy-v2/pull/40
* @oaazeved made their first contribution in https://github.com/MTgeophysics/mtpy-v2/pull/32

**Full Changelog**: https://github.com/MTgeophysics/mtpy-v2/compare/v2.0.7...v2.0.8

2.0.9 (2024-08-30)
----------------------

* Check for Pardiso import by @kujaku11 in #48

2.0.10 (2024-09-30)
---------------------

* Update testing_pip_import.yml by @kkappler in https://github.com/MTgeophysics/mtpy-v2/pull/49
* Kk/patches  by @kkappler in https://github.com/MTgeophysics/mtpy-v2/pull/53
* Optimize adding TF to MTCollection by @kujaku11 in https://github.com/MTgeophysics/mtpy-v2/pull/52
* @kkappler made their first contribution in https://github.com/MTgeophysics/mtpy-v2/pull/49

**Full Changelog**: https://github.com/MTgeophysics/mtpy-v2/compare/v2.0.9...v2.0.10

2.0.11 (2024-10-14)
------------------------

* Fix rotations again by @kujaku11 in https://github.com/MTgeophysics/mtpy-v2/pull/57
* Pin numpy versions <2.0 by @kkappler in https://github.com/MTgeophysics/mtpy-v2/pull/62
* Occam2d fixes by @alkirkby in https://github.com/MTgeophysics/mtpy-v2/pull/56
* Updates by @kujaku11 in https://github.com/MTgeophysics/mtpy-v2/pull/61

**Full Changelog**: https://github.com/MTgeophysics/mtpy-v2/compare/v2.0.10...v2.0.11

2.0.12 (2024-10-22)
----------------------------

* Fix rotations again by @kujaku11 in https://github.com/MTgeophysics/mtpy-v2/pull/57
* Pin numpy versions <2.0 by @kkappler in https://github.com/MTgeophysics/mtpy-v2/pull/62
* Occam2d fixes by @alkirkby in https://github.com/MTgeophysics/mtpy-v2/pull/56
* Updates by @kujaku11 in https://github.com/MTgeophysics/mtpy-v2/pull/61

**Full Changelog**: https://github.com/MTgeophysics/mtpy-v2/compare/v2.0.10...v2.0.12