# -*- coding: utf-8 -*-

"""Generic profiling context"""

__author__ = "Martin Böttcher, Brockmann Consult GmbH"
__copyright__ = "Copyright 2023, Brockmann Consult GmbH"
__license__ = "TBD"
__version__ = "0.5"
__email__ = "info@brockmann-consult.de"
__status__ = "Development"

# changes in 1.1:
# ...


class Profiling:

    def __init__(self, output: str = None):
        self._output = output

    def __enter__(self):
        if self._output:
            import cProfile

            self._profile = cProfile.Profile()
            self._profile.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._output:
            self._profile.disable()
            import io
            import pstats

            buffer = io.StringIO()
            stats = pstats.Stats(self._profile, stream=buffer)
            stats.sort_stats("tottime").print_stats()
            with open(self._output, "w") as f:
                f.write(buffer.getvalue())
