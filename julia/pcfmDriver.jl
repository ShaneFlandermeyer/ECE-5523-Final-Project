# using CSV
# using DelimitedFiles
using DSP
include("pcfm.jl")
include("profm.jl")

function write_binary_file(data,filename::String,dtype::DataType=Matrix{ComplexF32})
  f = open(filename,"w");
  data = convert(dtype,data)
  write(f,data)
  close(f)
end

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
    write_binary_file(s,"data/"*name*".bin")
    # sb = vcat(s, zeros(m-1,1))
    # sbf =  fftshift(fft(sb))
    # sbf = sbf ./maximum(abs.(sbf))
    # display(plot(10*log10.(abs.(sbf).^2),ylim=(-50, 0)))
    # display(plot!(10*log10.(u),ylim=(-50, 0)))
end

m = 1024
K=2
a = 10
iter = 1000
#Guassian Window
u = gaussian((2*m-1,1),0.1; padding = 0, zerophase = false)
exportAndDisplay(u,a,iter,K,m,"Gaussian")
#Hanning Window
#u = hanning((2*m-1,1); padding = 0, zerophase = false)
#exportAndDisplay(u,a,iter,K,m,"Hanning")
#Hamming Window
#u = hamming((2*m-1,1); padding = 0, zerophase = false)
#exportAndDisplay(u,a,iter,K,m,"Hamming")
#Tukey Window
# u = tukey((2*m-1,1), 0.5; padding = 0, zerophase = false)
# u[findall(<(-50), 10*log10.(u))] .= 10^-5
# u = abs.(u).^2
# time = @time begin
#   result = profm(sqrt.(u),iter)
# end
# time2 = @time begin
#   result2 = funPcfm(u,a,iter,K)
# end
# (B,Bb,x) = funPcfmHelper(m,K)
# s = exp.(im.*B*result2)
# sb = vcat(s, zeros(m-1,1))
# sbf =  fftshift(fft(sb))
# sbf = sbf ./maximum(abs.(sbf))
# resPlot = abs.(fftshift(fft(result)))
# resPlot = resPlot./maximum(resPlot)
# display(plot(abs.(resPlot).^2))

# display(plot!(abs.(sbf).^2))
# #exportAndDisplay(u,a,iter,K,m,"Tukey")
# #Rectangular Window

# #u = rect((2*m-1,1); padding = 0, zerophase = false)

# #exportAndDisplay(u,a,iter,K,m,"Rectangular")
# #Triangular Window
# #u = triang((2*m-1,1); padding = 0, zerophase = false)

# #exportAndDisplay(u,a,iter,K,m,"Triangular")