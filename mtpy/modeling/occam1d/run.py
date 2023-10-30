# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:35:16 2023

@author: jpeacock
"""

# =============================================================================
# Imports
# =============================================================================

# =============================================================================
class Run(object):
    """
    run occam 1d from python given the correct files and location of occam1d
    executable

    """

    def __init__(self, startup_fn=None, occam_path=None, **kwargs):
        self.startup_fn = startup_fn
        self.occam_path = occam_path
        self.mode = kwargs.pop("mode", "TE")

        self.run_occam1d()

    def run_occam1d(self):

        if self.startup_fn is None:
            raise IOError("Need to input startup file")
        if self.occam_path is None:
            raise IOError("Need to input path to occam1d executable")

        os.chdir(os.path.dirname(self.startup_fn))
        test = subprocess.call(
            [self.occam_path, os.path.basename(self.startup_fn), self.mode]
        )
        if test == 0:
            print("=========== Ran Inversion ==========")
            print(
                "  check {0} for files".format(os.path.dirname(self.startup_fn))
            )
