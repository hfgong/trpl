"""trpl_track: a NumPy/SciPy port of the ICCV 2011 multi-hypothesis
motion-planning tracker (Gong, Sim, Shi).

Pipeline: pretr -> filter -> appearance affinity -> motion-planning LP ->
finalize -> visualize.  See :mod:`trpl_track.run` for the end-to-end driver.
"""
from .config import DirectoryStructure, Parameters

__all__ = ["DirectoryStructure", "Parameters"]
__version__ = "0.1.0"
