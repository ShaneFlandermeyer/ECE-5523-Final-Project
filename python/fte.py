import numpy as np

import matplotlib.pyplot as plt

def fte(x, B, u):
    """
    Compute the frequency template error (FTE) metric from equation (5) of
    Mohr2018. This cost function tends to emphasize the passband of the template
    over the spectral roll-off region since the passband magnitude may be orders of
    magnitude higher than the roll-off region.

    Inputs:
      - x: An array of phase change values.
      - B: A matrix of basis functions, where each column corresponds to a shifted
           window function
      - u: The desired power spectrum 

    Outputs:
      - J: The FTE metric
      - sBar: The unshifted DFT of the phase coded waveform, padded to length 2M-1
      - sfBar: The shifted DFT of the phase coded waveform, padded to length 2M-1
    """
    # Phase coded waveform
    # x = np.array(x)
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
    # Compute FTE metric
    J = np.linalg.norm(abs(sfBar)**2-u,axis=0)
    return J
