# TODO: Rename this file

using Plots
using DSP
using FFTW
include("NoiseWaveform.jl")
"""
write_binary_file(data,filename,dtype)

# Arguments
- `data`: Data to be saved
- `filename::String`: Name of the file to be saved (path relative to current directory)
- `dtype::DataType`: Data type of the data to be saved
                     Default: `Matrix{ComplexF32}`
"""
function write_binary_file(data, filename::String, dtype::DataType=Matrix{ComplexF32})
    f = open(filename, "w")
    data = convert(dtype, data)
    write(f, data)
    close(f)
end

# PCFM Waveform Parameters
# ---------------
nWaveforms = 1
# Oversampled code length
m = 150
# Oversampling factor
k = 3
# Log base for log-FTE computation
a = 10

# Window Parameters
# -----------------
# Window length
l = 2 * m - 1
# Full-width 3dB (normalized) bandwidth
bandwidth = 0.15
sigma = bandwidth / (2 * sqrt(2 * log(2)))
# Window standard deviation
# sigma = 0.1
u = DSP.Windows.gaussian((l, 1), sigma)
# In practice, we probably won't be able to optimize the PSD below -50 dB, so
# clip all window values below it
u[findall(<(-50), 10 * log10.(u))] .= 10^-5

# Simulation Parameters
# ---------------------
maxIter = 1000
ϵ = 1e-5
(x, s) = NoiseWaveform.optimize(u, nWaveforms, k, a=10, tol=ϵ, maxIter=maxIter, showPlots=true)

write_binary_file(s, "data/gaussian.bin")