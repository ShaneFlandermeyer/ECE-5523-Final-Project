using FFTW
using LinearAlgebra
using DSP
using Plots
gr()
plot()
function ∇J(B,BBar,x,m,u,nt)
  s = exp.(im.*B*x)
  sb = vcat(s, zeros(m-1,1))
  sbf =  fftshift(fft(sb))
  sbf = sbf ./maximum(abs.(sbf))
  diff = abs.(sbf).^2 .-u
  J = norm(diff,Inf)

  return (J,2/J .*transpose(BBar)*imag.(conj.(sb).*ifft(ifftshift(diff.*sbf))))
end
function log∇J(B,BBar,x,m,u,nt,a)
  s = exp.(im.*B*x)
  sb = vcat(s, zeros(m-1,1))
  sbf =  fftshift(fft(sb))
  sbf = sbf ./maximum(abs.(sbf))
  diff = log.(a,abs.(sbf).^2) .-log.(a,u)
  J = norm(diff,2)
  
  return (J,2/(log(a)*J) .*transpose(BBar)*imag.(conj.(sb).*ifft(ifftshift(diff.*sbf))))
end

function funPcfmHelper(m,K)
  N::Int64 = trunc(m/K)
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
  BBar = vcat(B,zeros(m-1, nt))
  return (B,BBar,x)
end

function funPcfm(u)
  iter = 10000
  K =3
  m::Int64 = trunc((length(u)+1)/2)
  (B,BBar,x) = funPcfmHelper(m,K)
  epsilon = 10^-2
  i = 1
  μ = 0.25
  β = 0.5
  nt = m
  Jvec = ones(iter-1,1)
  vtOld = 0;
  while true
    (J,∇)=∇J(B,BBar,x.-β.*vtOld,m,u,nt)
    Jvec[i:end] .= J
    vt = β.*vtOld.+μ.*∇
    x -= vt
    s = exp.(im.*B*x)
    sb = vcat(s, zeros(m-1,1))
    sbf =  fftshift(fft(sb))
    sbf = sbf ./maximum(abs.(sbf))
    vtOld = vt
    if (i == iter )
      return x
    end
    display(plot(10*log10.(abs.(sbf).^2),ylim=(-50, 0)))
    display(plot!(10*log10.(u),ylim=(-50, 0)))
    #display(plot(Jvec))
    i += 1
  end
end

function funLogPcfm(u,a)
  iter = 1000
  K =4
  m::Int64 = trunc((length(u)+1)/2)
  (B,BBar,x) = funPcfmHelper(m,K)
  epsilon = 10^-2
  i = 1
  μ = 1
  β = 1
  nt = m
  Jvec = ones(iter-1,1)
  vtOld = 0;
  while true
    (J,∇)=log∇J(B,BBar,x.-β.*vtOld,m,u,nt,a)
    #Jvec[i:end] .= J
    vt = β.*vtOld.+μ.*∇
    x -= vt
    s = exp.(im.*B*x)
    sb = vcat(s, zeros(m-1,1))
    sbf =  fftshift(fft(sb))
    sbf = sbf ./maximum(abs.(sbf))
    Jvec[i:end] .= norm(abs.(sbf).^2 .-u)
    vtOld = vt
    if (i == iter )
      return x
    end
    
    display(plot(10*log10.(abs.(sbf).^2),ylim=(-60, 0)))
    display(plot!(10*log10.(u),ylim=(-60, 0)))
    #display(plot(Jvec))
    i += 1
  end
end

m = 128
u = abs.(gaussian((2*m-1,1),0.1; padding = 0, zerophase = false)).^2
u[findall(<(-50), 10*log10.(u))] .= 10^-5

#u = ones(64, 1)
#u = vcat(u, zeros(32-1,1).+0.001)
#u = vcat(zeros(32,1).+0.001,u)
display(plot(10*log10.(u),ylim=(-50, 0)))
time = @elapsed begin
   result = funLogPcfm(u,10)
end



println("Act time: ", time)
iter = 1000
println("Time per iter: ", time / iter)
