"""Top-level package for ras2d-viz."""

__author__ = """Christain Loving[D[D[D[D[D[D[Dia[1;5C[1;5C[1;5C[1;5C[1;5C[1;5C[1;5C[1;5C[1;5C[1;5C"""
__email__ = 'christian70401@gmail.com'
__version__ = '0.1.0'

#drill down to the goods no matter where you're importing from
if __package__ is None or __package__ == '':
    # uses current directory visibility
    from ras2d_viz.ras2d_viz import *
else:
    # uses current package visibility
    from .ras2d_viz.ras2d_viz import *