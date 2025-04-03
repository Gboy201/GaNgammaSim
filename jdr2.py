import numpy as np
from phase1 import depletionWidth, builtInPotential

# Constants
q = 1.6e-19  # Elementary charge
perimittivity = 8.9  # Permittivity
absolutePermittivity = perimittivity * 8.85e-12  # Absolute permittivity

def calculate_built_in_potential(temp, nSide, pSide):
    """Calculate the built-in potential for the junction."""
    return builtInPotential(temp, nSide, pSide)

def ElecFieldAsX(x, nSide, pSide, iSide, xp, W, xn):
    """Calculate the electric field at position x for a semiconductor junction.
    
    Args:
        x: Position (cm)
        nSide: Donor concentration (cm^-3)
        pSide: Acceptor concentration (cm^-3)
        iSide: Intrinsic concentration (cm^-3)
        xp: p-region width (cm)
        W: Depletion region width (cm)
        xn: n-region width (cm)
    
    Returns:
        float: Electric field value at position x (V/cm)
    """
    if x < 0:
        return ((-q*nSide)/absolutePermittivity)*(x+xn)
    if x >= 0 and x <= W:
        return (((-q*nSide)/absolutePermittivity)*(xn))-(((q*iSide)/absolutePermittivity)*x)
    if x > W:
        return ((q*pSide)/absolutePermittivity)*(x-(W+xp)) 