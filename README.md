# MTpy-v2: A Python Toolbox for working with Magnetotelluric (MT) Data

[![codecov](https://codecov.io/gh/MTgeophysics/mtpy-v2/graph/badge.svg?token=TQPFBFMYDQ)](https://codecov.io/gh/MTgeophysics/mtpy-v2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Description

A Python toolbox for working with magnetotelluric (MT) data, usually in the form of transfer functions, including

    - Transfer function analysis (plotting, strike analysis, phase tensors, etc.)
    - Write files for time series processing (BIRRP, Aurora, etc.)
    - Read/Write files for modeling programs (ModEM, Occam, Mare2DEM, etc)
    - Utility functions for GIS


**Note**: that this repository has superseded the [geophysics/mtpy](https://github.com/geophysics/mtpy/tree/beta)
and [GeoscienceAustralia/mtpy2](https://github.com/GeoscienceAustralia/mtpy2/tree/develop) and is an upgrade to [MTgeophysics/mtpy](https://github.com/MTgeophysics/mtpy).

# What's been Updated

The main updates in `mtpy-v2` are:

  - Remove dependence on EDI files, can be any type of transfer function file
      - Supports (or will support) to/from:
          - **EDI** (most common format)
          - **ZMM** (Egberts EMTF output)
          - **JFILE** (BIRRP output)
          - **EMTFXML** (Kelbert's format)
          - **AVG** (Zonge output)
  - Uses [mt-metadata](https://github.com/kujaku11/mt_metadata>) to read and write transfer function files where the transfer function data are stored in an [xarray](https://docs.xarray.dev/en/stable/index.html)
  - The workflow is more centralized by introducing `MTCollection` and `MTData` objects which are the databases to hold a collection of transfer functions and manipulate them
    - Includes plotting methods, `to/from` data file types for modeling, rotations, interpolations, static shifts, etc.
	- Can store a collection as an MTH5 using [mth5](https://github.com/kujaku11/mth5)

# Quick Example

Typically MT data are collected as surveys and each station produces a single transfer function.  These are provided in various formats like EDI, EMTF XML, etc.

One benefit of mtpy-v2 is reading all these in only needs to be done once and places them in a single MTH5 file.

```
from pathlib import Path
from mtpy import MTCollection

transfer_function_path = Path("/home/survey_00/transfer_functions")

# write directly to an MTH5 file and close when finished loading TFs
with MTCollection() as mc:
    mc.open_collection(transfer_function_path.joinpath("tf_collection.h5"))
    mc.add_tf(
        mc.make_file_list(
            transfer_function_path,
            file_types=["edi", "xml", "j", "zmm", "zss", "avg"],
        )
    )
 
```

Now when you want to access your data again, you just need to open a single file.

```
mc = MTCollection()
mc.open_collection(r"/home/survey_00/transfer_functions/tf_collection.h5")

# plot station locations
station_locations = mc.plot_stations()
```

# How to Cite

If you use this software in a scientific publication, we'd very much appreciate if you could cite the following papers:

- Kirkby, A.L., Zhang, F., Peacock, J., Hassan, R., Duan, J., 2019. The MTPy software package for magnetotelluric data analysis and visualisation. Journal of Open Source Software, 4(37), 1358. https://doi.org/10.21105/joss.01358
   
- Krieger, L., and Peacock, J., 2014. MTpy: A Python toolbox for magnetotellurics. Computers and Geosciences, 72, p167-175. https://doi.org/10.1016/j.cageo.2014.07.013



# Contacts

| **Jared Peacock**
| peacock.jared@gmail.com

| **Alison Kirkby**
| alkirkby@gmail.com


# System Requirements

-  Python 3.8+


# License

MTpy is licensed under the MIT license

The license agreement is contained in the repository and should be kept together with the code.


