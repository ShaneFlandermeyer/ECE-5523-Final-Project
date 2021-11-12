using FFTW
using LinearAlgebra
using DSP
using Plots
include("pcfm.jl")

function ∇J(B, x, u, l)
  """
  ∇J(B,Bb,x,m,u,a)
  Compute the frequency template error (FTE) and its associated gradient

  # Arguments
  - `B::Array`: M X M Orthogonal basis
  - `x::Vector`: M X 1 phase code vector
  - `u::Vector`: 2M-1 X 1 Frequency template  
  - `l::Integer`: Norm to use for error calculation
  """
  # Oversampled phase code length
  m = size(B, 1)
  # Zero-pad basis matrix to length 2M-1 (for FFT)
  Bb = vcat(B, zeros(m - 1, trunc(Int, m / k)))
  # PCFM representation of the input phase code vector
  s = exp.(im .* B * x)
  # Pad the waveform to length 2M-1
  sb = vcat(s, zeros(m - 1, 1))
  # Compute the (normalized) PSD of the PCFM waveform
  sbf = fftshift(fft(sb))
  sbf = sbf ./ maximum(abs.(sbf))
  # FTE calculation
  J = norm(abs.(sbf) .^ 2 .- u, l)
  # Return the error and gradient
  return (J,
    2 / (J) .* transpose(Bb) * imag.(conj.(sb) .* ifft(ifftshift((abs.(sbf) .^ 2 .- u) .* sbf))))
end

function ∇logJ(B, x, u, a, l)
  """
  ∇logJ(B,Bb,x,m,u,a)
  Compute the log frequency template error (log-FTE) and its associated gradient

  # Arguments
  - `B::Array`: M X M Orthogonal basis
  - `x::Vector`: M X 1 phase code vector
  - `u::Vector`: 2M-1 X 1 Frequency template  
  - `a::Integer`: Log base for error computation
  - `l::Integer`: Norm to use for error calculation

  """
  # Oversampled phase code length
  m = size(B, 1)
  # Zero-pad basis matrix to length 2M-1 (for FFT)
  Bb = vcat(B, zeros(m - 1, trunc(Int, m / k)))
  # PCFM representation of the input phase code vector
  s = exp.(im .* B * x)
  # Pad the waveform to length 2M-1
  sb = vcat(s, zeros(m - 1, 1))
  # Compute the (normalized) PSD of the PCFM waveform
  sbf = fftshift(fft(sb))
  sbf = sbf ./ maximum(abs.(sbf))
  # log-FTE calculation
  J = norm(log.(a, abs.(sbf) .^ 2) .- log.(a, u), l)
  # Return the error and gradient
  return (J,
    (2 / (log(a) * J)) .* transpose(Bb) * imag.(conj.(sb) .* ifft(ifftshift((log.(a, abs.(sbf) .^ 2) .- log.(a, u)) .* sbf))))
end

function profm(u, iter)
  """
  profm(u,iter)

  Iteratively optimize the PSD using alternating projections as described in the
  PRO-FM paper.
  """
  pk = exp.(im .* angle.(ifft(ifftshift(u))))
  for ii = 1:iter
    rk = ifft(ifftshift(abs.(u) .* exp.(im .* angle.(fftshift(fft(pk))))))
    pk = exp.(im .* angle.(rk))
    #display(plot(abs.(fftshift(fft(pk)))))
  end
  return pk
end

function optimize(u, a, k, tol, maxIter)
  """
  optimize(u,a,tol,maxIter)

  # Arguments
  - `u::Vector`: 2M-1 X 1 Frequency template
  - `a::Integer`: Log base for log-FTE (ignored if regular FTE is used)
  - `tol::Float`: Tolerance for early stopping
  - `maxIter::Integer`: Maximum number of iterations
  """
  #Calculate m from u.
  m = trunc(Int, (length(u) + 1) / 2)
  # Get a randomly initialized phase change vector and phase shaping basis
  # functions from PCFM generator
  (_, x, B) = pcfm(m, k)
  # Gradient descent Parameters
  μ = 0.75
  β = 0.9
  # Store the error at each iteration
  Jvec = ones(maxIter - 1, 1)
  pkOld = 0
  for ii = 1:maxIter
    # Heavy-ball gradient descent
    # (J, ∇) = ∇J(B, x, u, 2)
    (J, ∇) = ∇logJ(B, x, u, a, 2)
    Jvec[ii:end] .= J
    if ii == 1
      pk = ∇
    else
      pk = ∇ .+ β .* pkOld
    end
    xOld = x
    x -= μ .* pk
    if all(abs.(x .- xOld) .< tol)
      break
    end
    pkOld = pk

    #Extra Functions for visulation.
    s = exp.(im .* B * x)
    sb = vcat(s, zeros(m - 1, 1))
    sbf = fftshift(fft(sb))
    sbf = sbf ./ maximum(abs.(sbf))
    # corr = abs.(autocorr(s)) ./ maximum(abs.(autocorr(s)))
    # display(plot(10*log10.(corr),ylim=(-50,0)))
    display(plot!(10 * log10.(u), ylim = (-50, 0)))
    display(plot(10 * log10.(abs.(sbf) .^ 2), ylim = (-50, 0)))
  end
  return x

end

m = 150
k = 3
(s, alpha, B) = pcfm(m, k)
# Window function
u = gaussian((2 * m - 1, 1), 0.15; padding = 0, zerophase = false)
u[findall(<(-50), 10 * log10.(u))] .= 10^-5
u = abs.(u) .^ 2
a = 10
iter = 1000
optimize(u, a, k, 4e-4, iter)
