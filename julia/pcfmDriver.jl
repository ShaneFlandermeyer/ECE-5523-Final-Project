using CSV
using DelimitedFiles
include("pcfm.jl")
"""
    exportAndDisplay(u,a,iter,K,m,name)
Exports and displays PCFM algorith results.
  ...
# Arguments
- `u::Vector`: Window Function PSD.
- `a::Integer`: Log base. If zero use non log version.
- `iter::Integer`: Number of iterations.
- `K::Integer`: Oversampling factor.
- `m::Integer`: Size.
- `name::String`: Name of window function.
...
"""
function exportAndDisplay(u,a,iter,K,m,name)
    u[findall(<(-50), 10*log10.(u))] .= 10^-5
    u = abs.(u).^2
    result = funPcfm(u,a,iter,K)
    (B,Bb,x) = funPcfmHelper(m,K)
    s = exp.(im.*B*result)
    writedlm( name*".csv",  [real(s),imag(s)], ',')
    sb = vcat(s, zeros(m-1,1))
    sbf =  fftshift(fft(sb))
    sbf = sbf ./maximum(abs.(sbf))
    display(plot(10*log10.(abs.(sbf).^2),ylim=(-50, 0)))
    display(plot!(10*log10.(u),ylim=(-50, 0)))
end

m = 128
K=3
a = 10
iter = 10000
#Guassian Window
u = gaussian((2*m-1,1),0.1; padding = 0, zerophase = false)
exportAndDisplay(u,a,iter,K,m,"Guassian")
#Hanning Window
u = hanning((2*m-1,1); padding = 0, zerophase = false)
exportAndDisplay(u,a,iter,K,m,"Hanning")
#Hamming Window
u = hamming((2*m-1,1); padding = 0, zerophase = false)
exportAndDisplay(u,a,iter,K,m,"Hamming")
#Tukey Window
u = tukey((2*m-1,1), 0.5; padding = 0, zerophase = false)
exportAndDisplay(u,a,iter,K,m,"Tukey")
#Rectangular Window
u = rect((2*m-1,1); padding = 0, zerophase = false)
exportAndDisplay(u,a,iter,K,m,"Rectangular")
#Triangular Window
u = triang((2*m-1,1); padding = 0, zerophase = false)
exportAndDisplay(u,a,iter,K,m,"Triangular")