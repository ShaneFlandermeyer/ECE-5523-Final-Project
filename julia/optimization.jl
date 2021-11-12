using FFTW
using LinearAlgebra
using DSP
using Plots
include("pcfm.jl")

function ∇J(B,x,u,l)
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
  m = size(B,1)
  # Zero-pad basis matrix to length 2M-1 (for FFT)
  Bb = vcat(B,zeros(m-1,trunc(Int,m/k)))
  # PCFM representation of the input phase code vector
  s = exp.(im.*B*x)
  # Pad the waveform to length 2M-1
  sb = vcat(s, zeros(m-1,1))
  # Compute the (normalized) PSD of the PCFM waveform
  sbf =  fftshift(fft(sb))
  sbf = sbf ./maximum(abs.(sbf))
  # FTE calculation
  J = norm(abs.(sbf).^2 .-u,l)
  # Return the error and gradient
  return (J,
  2/(J).*transpose(Bb)*imag.(conj.(sb).*ifft(ifftshift((abs.(sbf).^2 .-u).*sbf))))
end

function ∇logJ(B,x,u,a,l)
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
  m = size(B,1)
  # Zero-pad basis matrix to length 2M-1 (for FFT)
  Bb = vcat(B,zeros(m-1,trunc(Int,m/k)))
  # PCFM representation of the input phase code vector
  s = exp.(im.*B*x)
  # Pad the waveform to length 2M-1
  sb = vcat(s, zeros(m-1,1))
  # Compute the (normalized) PSD of the PCFM waveform
  sbf =  fftshift(fft(sb))
  sbf = sbf ./maximum(abs.(sbf))
  # log-FTE calculation
  J = norm(log.(a,abs.(sbf).^2) .-log.(a,u),l)
  # Return the error and gradient
  return (J,
  (2/(log(a)*J)).*transpose(Bb)*imag.(conj.(sb).*ifft(ifftshift((log.(a,abs.(sbf).^2) .-log.(a,u)).*sbf))))
end


function funPcfm(u,a,iter,K)
  """
  funPcfm(u,a,iter,K)
  PCFM algorithm. Returns a randomly initialized vector with PSD of u.
    ...
  # Arguments
  - `u::Vector`: Window Function PSD.
  - `a::Integer`: Log base. If zero do non log version.
  - `iter::Integer`: Number of iterations.
  - `K::Integer`: Oversampling Factor.
  ...
  """
  #Calculate m from u.
  m = trunc(Int,(length(u)+1)/2)
  (s,x,B) = pcfm(m,K)
  #Gradient Descent Parameters
  μ = 0.75
  β = 0.5
  #Vector of error at each iteration.
  Jvec = ones(iter-1,1)
  i = 1
  vtOld = 0;
  while i <= iter
    #Nesterov Accelerated Descent
    # (J,∇)=∇J(B,x.-β.*vtOld,u,2)
    (J,∇) = ∇logJ(B,x.-β.*vtOld,u,a,2)
    Jvec[i:end] .= J
    vt = β.*vtOld.+μ.*∇
    x -= vt
    vtOld = vt

    #Extra Functions for visulation.
    s = exp.(im.*B*x)
    sb = vcat(s, zeros(m-1,1))
    sbf =  fftshift(fft(sb))
    sbf = sbf ./maximum(abs.(sbf))
    display(plot!(10*log10.(u)))
    display(plot(10*log10.(abs.(sbf).^2),ylim=(-50,0)))
    
    # corr = abs.(autocorr(s)) ./ maximum(abs.(autocorr(s)))
    # display(plot(10*log10.(corr),ylim=(-50,0)))
    i += 1
  end
  return x
end

m = 150
k = 3
(s,alpha,B) = pcfm(m,k)
# Window function
u = gaussian((2*m-1,1),0.1; padding = 0, zerophase = false)
u[findall(<(-50), 10*log10.(u))] .= 10^-5
u = abs.(u).^2
a = 0
iter = 1000
funPcfm(u,a,iter,k)
