using FFTW
using LinearAlgebra
using DSP
using Plots
gr()
plot()
function autocorr(x)
  return conv(x,x[end:-1:1])
end
function ∇J(B,Bb,x,m,u,a)
  s = exp.(im.*B*x)
  sb = vcat(s, zeros(m-1,1))
  sbf =  fftshift(fft(sb))
  sbf = sbf ./maximum(abs.(sbf))
  if(a == 0)
    diff = abs.(sbf).^2 .-u
    J = norm(diff,2)
    return (J,2/J .*transpose(Bb)*imag.(conj.(sb).*ifft(ifftshift(diff.*sbf))))
  else
    diff = log.(a,abs.(sbf).^2) .-log.(a,u)
    J = norm(diff,2)
    return (J,2/(log(a)*J) .*transpose(Bb)*imag.(conj.(sb).*ifft(ifftshift(diff.*sbf))))
  end
end
function funPcfmHelper(m,K)
  nt = m
  minAlpha = -pi/K
  maxAlpha = pi/K
  g = ones(K, 1)./K
  g = vcat(g, zeros(m-K,1))
  B = cumsum(g, dims = 1)
  for i = 2:nt
    g = vcat(0, g[1:end-1])
    B = hcat(B, cumsum(g, dims = 1))
  end
  x = minAlpha .+(maxAlpha-minAlpha).*rand(Float64,(nt, 1))
  Bb = vcat(B,zeros(m-1, nt))
  return (B,Bb,x)
end
function funPcfm(u,a,iter,K)
  m::Int64 = trunc((length(u)+1)/2)
  (B,Bb,x) = funPcfmHelper(m,K)
  epsilon = 10^-2
  μ = 0.25
  β = 0.5
  Jvec = ones(iter-1,1)
  i = 1
  vtOld = 0;
  while i <= iter
    (J,∇)=∇J(B,Bb,x.-β.*vtOld,m,u,a)
    Jvec[i:end] .= J
    vt = β.*vtOld.+μ.*∇
    x -= vt
    vtOld = vt

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
