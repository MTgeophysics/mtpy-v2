.. highlight:: shell

============
Installation
============


Stable release
--------------

PIP
^^^^

To install `mt_metadata`, run this command in your terminal:

.. code-block:: console

    $ pip install mtpy-v2

This is the preferred method to install mt_metadata, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/

Conda-Forge
^^^^^^^^^^^^^
To install `mtpy-v2`, run either of these commands in your Conda terminal (`<https://conda-forge.org/#about>`_):

.. code-block:: console
    
	$ conda install -c conda-forge mtpy-v2

or 

.. code-block:: console

    $ conda config --add channels conda-forge
    $ conda config --set channel_priority strict
    $ conda install mtpy-v2 


.. note:: If you are updating `mt_metadata` you should use the same installer as your previous version or remove the current version and do a fresh install. 

From sources
------------

The sources for MTH5 can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/MTgeophysics/mtpy-v2

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/MTgeophysics/mtpy-v2/tarball/main

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/MTgeophysics/mtpy-v2
.. _tarball: https://github.com/MTgeophysics/mtpy-v2/tarball/main
