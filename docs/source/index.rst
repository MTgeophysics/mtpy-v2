Welcome to MTpy-v2 documentation!
======================================

.. |License| image:: https://img.shields.io/github/license/Naereen/StrapDown.js.svg
   :target: https://github.com/Naereen/StrapDown.js/blob/master/LICENSE
.. |Codecov|image:: https://codecov.io/gh/MTgeophysics/mtpy-v2/graph/badge.svg?token=TQPFBFMYDQ 
 :target: https://codecov.io/gh/MTgeophysics/mtpy-v2


`mtpy` provides tools for working with magnetotelluric (MT) data.  MTpy-v2 is an update version of [mtpy](https://github.com/MTgeophysics/mtpy). Many things have changed under the hood and usage is different from mtpy v1. The main difference is that there is a central data type that can hold transfer functions and then read/write to your modeling program, plot, and analyze your data.  No longer will you need a directory of EDI files and then read them in everytime you want to do something.  You only need to build a project once and save it to an MTH5 file and you are ready to go. All metadata uses [mt-metadata](https://github.com/kujaku11/mt-metadata).  

Because the workflow has changed from mtpy v1, there are example notebooks to demonstrate the new workflow see :ref:`ref-usage`.  

Examples
-------------

.. |Binder| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/MTgeophysics/mtpy-v2/main
 
 
Click on the `Binder` badge above to interact with Jupyter Notebook examples.  There are example notebooks in
    
	- **docs/source/examples/notebooks** 

.. toctree::
    :maxdepth: 1
    :caption: General Information

    installation
    usage
    contributing
    authors
    history
    conventions   

.. toctree::
    :maxdepth: 1
    :caption: API Reference
    
    modules
    

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
