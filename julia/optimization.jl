using FFTW
using LinearAlgebra
using DSP
using Plots
include("pcfm.jl")

function ∇J(B,Bb,x,m,u,a)
  """
    ∇J(B,Bb,x,m,u,a)
  Compute the gradient for the PCFM algorithm.
    ...
  # Arguments
  - `B::Array`: M X M Orthogonal basis.
  - `Bb::Array`: 2M-1 X M Orthogonal basis.
  - `x::Vector`: M X 1 Phase coefficients.
  - `m::Integer`: Size.
  - `u::Vector`: 2M-1 X 1 Window Function.
  - `a::Integer`: Log base. If zero than use non-log version.
  ...
  """
  s = exp.(im.*B*x)
  sb = vcat(s, zeros(m-1,1))
  sbf =  fftshift(fft(sb))
  sbf = sbf ./maximum(abs.(sbf))
  if(a == 0)
    diff = abs.(sbf).^2 .-u
    factor = 2
  else
    diff = log.(a,abs.(sbf).^2) .-log.(a,u)
    factor = 2/(log(a))
  end
  J = norm(diff,2)
  #Need shift since PSD is centered about 0.
  return (J,factor/(J).*transpose(Bb)*imag.(conj.(sb).*ifft(ifftshift(diff.*sbf))))
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
  Bb = vcat(B,zeros(m-1, trunc(Int,m/k)))
  # (B,Bb,x) = funPcfmHelper(m,K)
  #Gradient Descent Parameters
  μ = 0.75
  β = 0.5
  #Vector of error at each iteration.
  Jvec = ones(iter-1,1)
  i = 1
  vtOld = 0;
  while i <= iter
    #Nesterov Accelerated Descent
    (J,∇)=∇J(B,Bb,x.-β.*vtOld,m,u,a)
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
u = gaussian((2*m-1,1),0.15; padding = 0, zerophase = false)
u[findall(<(-50), 10*log10.(u))] .= 10^-5
u = abs.(u).^2
a = 0
iter = 1000
funPcfm(u,a,iter,k)
