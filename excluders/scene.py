import os
import re

import numpy as np


class Excluder:
    """
    we need to exclude both the different scene (SID)
    """
    def __init__(self, gallery_fids):
        # Setup a regexp for extracing the PID and camera (CID) form a FID.
        self.regexp = re.compile('(\S+)_c(\d+)s(\d+)_.*')

        # Parse the gallery_set
        self.gallery_sids = self._parse(gallery_fids)

    def __call__(self, query_fids):
        # Extract both the PIDs and CIDs from the query filenames:
        query_sids = self._parse(query_fids)

        # Ignore different pid image within different scenes
        return self.gallery_sids[None] != query_sids[:,None]

    def _parse(self, fids):
        """ Return the SIDs extracted from the FIDs. """
        sids = []
        for fid in fids:
            filename = os.path.splitext(os.path.basename(fid))[0]
            norm_filename = os.path.normpath(fid)
            dirname = norm_filename.split(os.sep)[0]
            sids.append(dirname)
        return np.asarray(sids)
