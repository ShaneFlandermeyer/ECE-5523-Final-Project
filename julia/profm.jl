
function profm(u,iter)
    pk = exp.(im.*angle.(ifft(ifftshift(u))))
    for i = 1:iter
        rk = ifft(ifftshift(abs.(u).*exp.(im.*angle.(fftshift(fft(pk))))))
        pk = exp.(im.*angle.(rk))
        #display(plot(abs.(fftshift(fft(pk)))))
    end
    return pk
end