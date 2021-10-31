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
    J = fte(x, B, u)
    s = np.exp(1j*B@x)
    if s.ndim == 1:
      s = s[:, np.newaxis]
    if u.ndim == 1:
      u = u[:,np.newaxis]
      
    # Length of the phase-coded waveform
    M = s.shape[0]
    # Compute the length-2M-1 DFT of the phase coded waveform
    sBar = np.concatenate((s, np.zeros((M-1, s.shape[1]))))
    # 
    sfBar = np.fft.fftshift(np.fft.fft(sBar,axis=0))
    sfBar = sfBar / max(abs(sfBar))
    
    # Pad B to account for the zero padding in sBar
    M, Ntilde = B.shape
    BBar = np.concatenate((B.T, np.zeros((Ntilde, M-1))),axis=1).T
    # Gradient computation
    grad = 2/J*BBar.T@np.imag(
      np.conj(sBar) * ifft(ifftshift((abs(sfBar)**2-u)*sfBar),axis=0))
    return grad
