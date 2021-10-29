import numpy as np
from numpy.fft import ifft, ifftshift
from fte import fte


def gradFte(x, B, u):
    """
    Computes the gradient of the FTE function.

    Inputs:
    - x: An array of phase change values.
    - B: A matrix of basis functions, where each column corresponds to a shifted
         window function
    - u: The desired power spectrum 

  Outputs:
    - gradF: The FTE gradient with respect to the current x
    """
    J, sBar, sfBar = fte(x, B, u)
    # Pad B to account for the zero padding in sBar
    M, Ntilde = B.shape
    BBar = np.concatenate((B.T, np.zeros((Ntilde, M-1))), axis=1).T
    # Gradient computation
    return 2/J*BBar.T@np.imag(
      np.conj(sBar) * ifft(ifftshift(((np.power(sfBar, 2)-u[:, np.newaxis])*sfBar))))
