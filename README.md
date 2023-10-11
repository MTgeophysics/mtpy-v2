# MTpy-v2: A Python Toolbox for Magnetotelluric (MT) Data Processing, Analysis, Modelling and Visualization

|Documentation Status|

# Overview

A Python Toolbox for Magnetotelluric (MT) Data Processing, Analysis, Modelling and Visualization

- Home Page: https://github.com/MTgeophysics/mtpy

- API Documentation: http://mtpy-v2.readthedocs.io/en/develop/

- Issue tracking: https://github.com/MTgeophysics/mtpy-v2/issues

- Installation Guide (Wiki Pages): https://github.com/MTgeophysics/mtpy-v2/wiki

Note that this repository has superseded the [geophysics/mtpy](https://github.com/geophysics/mtpy/tree/beta)
and [GeoscienceAustralia/mtpy2](https://github.com/GeoscienceAustralia/mtpy2/tree/develop) and is an upgrade to [MTgeophysics/mtpy](https://github.com/MTgeophysics/mtpy).

# Updated

The main updates in `mtpy-v2` are:

  - Remove dependence on just a group of EDI files, can be any type of transfer function file
  - Use [mt-metadata](https://github.com/kujaku11/mt_metadata>) to read and write transfer function files where the transfer function data are stored in an [xarray](https://docs.xarray.dev/en/stable/index.html)
  - The workflow is more centralized by introducing `MTCollection` and `MTData` objects which are the databases to hold a collection of transfer functions and manipulate them
    - Includes plotting methods, `to/from` data file types for modeling, rotations, interpolations, static shifts, etc.
	- Can store a collection as an MTH5 using [mth5](https://github.com/kujaku11/mth5)
  - 

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

MTpy is licensed under the GPL version 3

The license agreement is contained in the repository and should be kept together with the code.

