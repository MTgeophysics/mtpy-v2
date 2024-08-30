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

## New Contributors
* @oaazeved made their first contribution in https://github.com/MTgeophysics/mtpy-v2/pull/32

**Full Changelog**: https://github.com/MTgeophysics/mtpy-v2/compare/v2.0.7...v2.0.8