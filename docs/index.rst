Welcome to MT Metadata documentation!
======================================

.. image:: https://img.shields.io/pypi/v/mt_metadata.svg
        :target: https://pypi.python.org/pypi/mt_metadata

.. image:: https://img.shields.io/conda/v/conda-forge/mt-metadata.svg
    :target: https://anaconda.org/conda-forge/mt-metadata
    :alt: Latest conda-forge version
	
.. image:: https://readthedocs.org/projects/mt-metadata/badge/?version=latest
    :target: https://mt-metadata.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
		
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://github.com/kujaku11/mt_metadata/blob/main/LICENSE
    :alt: MIT license

.. image:: https://codecov.io/gh/kujaku11/mt_metadata/branch/main/graph/badge.svg?token=XU5QSRM1ZO
        :target: https://codecov.io/gh/kujaku11/mt_metadata
        
.. image:: https://zenodo.org/badge/283883448.svg
   :target: https://zenodo.org/badge/latestdoi/283883448
   
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/kujaku11/mt_metadata/main

MT Metadata is meant to be a tool to standardize both time series and transfer function metadata for magnetotelluric data.

The base metadata object is structured in a flexible way to accommodate any type of metadata, you just need to formulate the standards following the format described in the documentation.  See :ref:`structure` and :ref:`ref-usage` for more details. 

Examples
-------------

Click on the `Binder` badge above to interact with Jupyter Notebook examples.  There are examples in
    
	- **docs/source/notebooks**
	- **mt_metadata/examples/notebooks**  

.. toctree::
    :maxdepth: 1
    :caption: General Information

    readme
    installation
    usage
    contributing
    authors
    history
    conventions
    source/notebooks/ts_metadata_examples.ipynb
    
.. toctree::
    :maxdepth: 1
    :caption: Basics of Metadata Structure
    
    source/structure
    
.. toctree::
    :maxdepth: 1
    :caption: Time Series

    source/ts_metadata_guide
    source/notebooks/filters_example.ipynb
    
.. toctree::
    :maxdepth: 1
    :caption: Transfer Functions
    
    source/tf_structure
    source/tf_processing_aurora_index
    source/tf_processing_fcs_index
        

.. toctree::
    :maxdepth: 1
    :caption: API Reference
    
    source/modules
    

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
