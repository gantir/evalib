# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version

    from . import utils
    from . import plot
    from . import data
    from . import datasets
    from . import models
    from . import gradcam
    from . import tnt
    from . import lr

    __all__ = ["utils", "plot", "data", "models", "gradcam", "tnt"]
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
