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

