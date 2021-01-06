import os
import re

import numpy as np


class Excluder:
    """
    we need to exclude both the different scene (SID)
    """
    def __init__(self, gallery_fids, gallery_rids):
        # Setup a regexp for extracing the PID and camera (CID) form a FID.
        self.regexp = re.compile('(\S+)_c(\d+)s(\d+)_.*')

        # Parse the gallery_set
        self.gallery_sids = gallery_rids

    def __call__(self, query_fids, query_rids):
        # Extract both the PIDs and CIDs from the query filenames:
        query_sids = self._parse(query_rids)

        # Ignore different pid image within different scenes
        return self.gallery_sids[None] != query_sids[:,None]

    def _parse(self, rids):
        """ Return the SIDs extracted from the FIDs. """
        sids = []
        for rid in rids:
            filename = os.path.splitext(os.path.basename(rid))[0]
            norm_filename = os.path.normpath(rid)
            dirname = norm_filename.split(os.sep)[0]
            sids.append(dirname)
        return np.asarray(sids)
