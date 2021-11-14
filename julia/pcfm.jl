using FFTW
using LinearAlgebra
using DSP
using Plots

function pcfm(m,k)
  """
  pcfm(m,k)
  Compute the PCFM waveform for a randomly initialized phase code

    # Arguments
    - `m::Integer`: Phase code length
    - `k::Integer`: Oversampling factor
  """
  # Number of unique phase code values (i.e., no oversampling)
  n = trunc(Int,m/k)
  # Construct the array of phase code changes
  minAlpha = -pi/k
  maxAlpha = pi/k
  alpha = minAlpha .+(maxAlpha-minAlpha).*rand(Float64,(n, 1))
  # Phase shaping filter
  g = ones(k,1)./k
  g = vec(vcat(g,zeros(m-k,1)))
  B = zeros(m,n)
  for ii in 1:n
    B[:,ii] = cumsum(shiftsignal(g,ii*k))
  end
  s = exp.(im.*B*alpha)
  return (s,alpha,B)
end
