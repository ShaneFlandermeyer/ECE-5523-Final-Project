using FFTW
using LinearAlgebra
using DSP
using Plots
gr()
plot()
function ∇J(B,x,m,u,nt)
  s = exp.(im.*B*x)
  sb = vcat(s, zeros(m-1,1))
  sbf =  fftshift(fft(sb))
  sbf = sbf ./maximum(abs.(sbf))
  J = norm(abs.(sbf).^2 .-u)
  BBar = vcat(B,zeros(m-1, nt))
  return (J,2/J .*transpose(BBar)*imag.(conj.(sb).*ifft(ifftshift((abs.(sbf).^2 .-u).*sbf))))
end
function funPcfmHelper(m,K)
  N::Int64 = trunc(m/K)
  minAlpha = -pi/K
  maxAlpha = pi/K
  x = minAlpha .+(maxAlpha-minAlpha).*rand(Float64,(N, 1))
  g = ones(K, 1)./K
  g = vcat(g, zeros(m-K,1))
  B = cumsum(g, dims = 1)
  for i = 2:N
    g = vcat(0, g[1:end-1])
    B = hcat(B, cumsum(g, dims = 1))
  end
  (m, nt) = size(B)
  BBar = vcat(B,zeros(m-1, nt))
  return (B,BBar,x)
end

function funPcfm(u)
  iter = 10^8
  K =1
  m::Int64 = trunc((length(u)+1)/2)
  (B,BBar,x) = funPcfmHelper(m,K)
  epsilon = 10^-2
  i = 1
  μ = 0.25
  β = 0.5
  nt = m
  Jvec = ones(iter,1)
  vtOld = 0;

  while true
    (J,∇)=∇J(B,x.-β.*vtOld,m,u,nt)
    Jvec = i == 1 ? J.*Jvec : Jvec;
    Jvec[i] = J
    vt = β.*vtOld.+μ.*∇
    x -= vt
    #s = exp.(im.*B*x)
    #sb = vcat(s, zeros(m-1,1))
    #sbf =  fftshift(fft(sb))
    #sbf = sbf ./maximum(abs.(sbf))
    vtOld = vt
    if (i == iter || norm(vt) < epsilon)
      return x
    end
    #display(plot(abs.(sbf).^2))
    i += 1
  end
end

m = 2^10
u = abs.(gaussian((2*m-1,1),0.1; padding = 0, zerophase = false)).^2

time = @elapsed begin
   result = funPcfm(u)
end

#display(plot!(u))

println("Act time: ", time)
iter = 10^8
println("Time per iter: ", time / iter)
