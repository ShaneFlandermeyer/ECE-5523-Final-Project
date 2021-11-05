using FFTW
using LinearAlgebra
using DSP
using Plots
gr()
plot()
function autocorr(x)
  return conv(x,x[end:-1:1])
end
"""
    ∇J(B,Bb,x,m,u,a)
Compute the gradient for the PCFM algorithm.
  ...
# Arguments
- `B::Array`: M X M Orthonormal basis.
- `Bb::Array`: 2M-1 X M Orthonormal basis.
- `x::Vector`: M X 1 Phase coefficients.
- `m::Integer`: Size.
- `u::Vector`: 2M-1 X 1 Window Function.
- `a::Integer`: Log base. If zero than use non-log version.
...
"""
function ∇J(B,Bb,x,m,u,a)
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
"""
  funPcfmHelper(m,K)
Computes extra values for use in gradient calculation.
  ...
# Arguments
- `m::Integer`: Size.
- `K::Integer`: Oversampling Factor.
...
"""
function funPcfmHelper(m,K)
  nt = m
  minAlpha = -pi/K
  maxAlpha = pi/K
  #Hard way of making a ramp function.
  g = ones(K, 1)./K
  g = vcat(g, zeros(m-K,1))
  B = cumsum(g, dims = 1)
  for i = 2:nt
    #Logical Shift g
    g = vcat(0, g[1:end-1])
    B = hcat(B, cumsum(g, dims = 1))
  end
  x = minAlpha .+(maxAlpha-minAlpha).*rand(Float64,(nt, 1))
  Bb = vcat(B,zeros(m-1, nt))
  return (B,Bb,x)
end
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
function funPcfm(u,a,iter,K)
  #Calculate m from u.
  m = trunc(Int,(length(u)+1)/2)
  (B,Bb,x) = funPcfmHelper(m,K)
  #Gradient Descent Parameters
  μ = 0.25
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
    #Extra Functions fro visulation.
    #s = exp.(im.*B*x)
    #sb = vcat(s, zeros(m-1,1))
    #sbf =  fftshift(fft(sb))
    #sbf = sbf ./maximum(abs.(sbf))
    #display(plot(real((abs.(autocorr(sbf))))))
    #display(plot(10*log10.(abs.(sbf).^2),ylim=(-50, 0)))
    #display(plot!(10*log10.(u),ylim=(-50, 0)))
    #display(plot(Jvec))
    i += 1
  end
  return x
end