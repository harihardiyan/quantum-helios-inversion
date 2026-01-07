# __init__.py
# Author: Hari Hardiyan <lorozloraz@gmail.com>

from .model import PIQNN, QuantumLayer, OutputLogRangeAdaptor
from .physics import parker_u_at_r_supersonic, physics_residuals
from .utils import StandardScalerJAX

__version__ = "1.0.0"
__author__ = "Hari Hardiyan"

# Memungkinkan user memanggil: 
# from quantum_helios import PIQNN
