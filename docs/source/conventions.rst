============
Conventions
============

Some conventions that have been implemented:

	* Attribute names are all lower case and word separated by '_'
	* All times are UTC, other time zones are supported but strongly discouraged.
	* Units should be in SI and when a name is required should be all lower case and spelled out, for example 'nanotesla' or 'millivolt per kilometer'
	* All azimuth angles should be relative to geographic north and measured positive clockwise in a right-handed coordinate system with z positive downwards.
	  * NED+ is the default coordinate system x=North, y=East, z=Down. Other coordinate systems are supported but strongly discouraged. 
	  * ENU- is the other coordinate system x=East, y=North, z=Up. 
	* All rotation angles should be in degrees
	* Locations are given in latitude and longitude in decimal degrees and should all use the same well known datum.  Default is WGS84.
	* All metadata should be in English.
	* All metadata should be in UTF-8 encoding.