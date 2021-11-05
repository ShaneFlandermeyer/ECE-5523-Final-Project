% Implement the figures from Mohr2018 - Spectrally-Efficient FM Noise Radar
% Waveforms Optimized in the Logarithmic Domain
%
% Author: Shane Flandermeyer
%%
clc;close all force;clear;
%% PCFM waveform representation
% Number of waveforms to use
nWaveforms = 1;
% Waveform length (samples)
M = 900;
% Oversampling factor
K = 3;
% Number of code values
N = M/K;
% Randomly generate code sample matrix for each waveform
minAlpha = -pi/K; maxAlpha = pi/K;
x = minAlpha + (maxAlpha-minAlpha)*rand(N,nWaveforms);
% RectangularShaping filter with time support [0,Tp) = [0,K]
g = ones(K,1) ./ K;
g = [g;zeros(M-K,1)];
% Construct the M x N basis matrix where each column is a basis function.
% That is, each column is equal to the previous column delayed by K samples
% TODO: Find a better way to construct a delay matrix
B = zeros(M,N);
for ii = 1 : N
  B(:,ii) = cumsum(delayseq(g,(ii-1)*K));
end

%% Frequency Domain Metric
% Set the desired FULL-WIDTH 3dB (normalized) bandwidth
bw = 0.25;
% Convert to standard deviations
% https://en.wikipedia.org/wiki/Full_width_at_half_maximum
sigma = bw/(2*sqrt(2*log(2)));
% Convert from standard deviation to Matlab's weird width factor
% https://www.mathworks.com/help/signal/ref/gausswin.html#References
alpha = 1/2/sigma;
% Create the gaussian template
u = abs(gausswin(2*M-1,alpha)).^2;
% Clip the power spectrum at -50 dB
u(10*log10(u) < -50) = 0;
% Sample rate
fs = 200e6;
% Frequency axis for plotting
freqStep = 2*fs/(2*M-1);
freqAxis = (-fs:freqStep:fs-freqStep).';
[~,~,initialSbar] = fte(x,B,u);
plot(freqAxis,u)
%% Gradient-Based Optimization
% Function handles for the cost function and its gradient, where we assume
% the basis matrix B and spectral template u remain unchanged
f = @(x) fte(x,B,u);
gradF = @(x) gradFte(x,B,u);
pk = -gradF(x);
beta = 0.5;
nIterations = 1e4;
% Set the step size
J = zeros(nWaveforms,nIterations);
% J(1,ii) = fte(x,B,u);
waitBar = waitbar(0,sprintf('J = %0.1f',J(ii)));
% close(waitBar)
for ii = 1 : nIterations
  % Do heavy-ball gradient descent
  J(:,ii) = fte(x,B,u);
  waitbar(ii/nIterations,waitBar,sprintf('J = %0.1f',mean(J(:,ii))));
%   mu_k = backtracking_line_search(f,gradF,x,pk);
  mu_k = 0.25;
  x = x + mu_k*pk;
  if ii > 1 && J(:,ii) > J(:,ii-1)
    pk = -gradF(x);
  else
    pk = -gradF(x)+beta*pk;
  end
  
end
close(waitBar)
%% Plots
figure()
s = exp(1i*B*x);
sBar = [s.' zeros(nWaveforms,M-1)].';
sfBar = fftshift(fft(sBar));
sfBar = sfBar ./ max(sfBar);
plot(freqAxis,10*log10(abs(sfBar).^2))
hold on
plot(freqAxis,10*log10(abs(initialSbar).^2))
plot(freqAxis,10*log10(abs(u).^2))
hold off
legend('Optimized','Initial','Template')
figure()
plot(J)
%% Helper Functions
function [J,sBar,sfBar] = fte(x,B,u)
% Compute the frequency template error (FTE) metric from equation (5)
% This cost function tends to emphasize the passband of u over the spectral
% roll-off region since the passband may be orders of magnitude higher
%
% Inputs:
%   - x: The discretized phase change values
%   - B: A matrix of discretized basis functions, where each column
%        corresponds to a basis function
%   - u: The disretized representation of the desired power spectrum
% Outputs:
%   - J: FTE metric
% TODO: Allow for norms other than L2

% Phase coded waveform
s = exp(1i*B*x);
% Number of samples in the phase-coded waveform (Accounting for
% oversampling)
M = size(s,1);
% Compute a DFT of length 2M-1 for the phase coded waveform
sBar = [s; zeros(M-1,size(s,2))];
sfBar = fftshift(fft(sBar));
sfBar = sfBar ./ max(abs(sfBar));
% Compute FTE
J = norm(abs(sfBar).^2 - u);
end

function gradF = gradFte(x,B,u)
% Computes the gradient of the frequency template error (FTE) metric (eq. 12)
%
% Inputs:
%   - x: A discretized array of phase change values
%   - B: A matrix of discretized basis functions, where each column
%        corresponds to a basis function
%   - u: The disretized representation of the desired power spectrum
% Outputs:
%   - gradF: The FTE gradient with respect to the current x

[J,sBar,sfBar] = fte(x,B,u);
% Pad B to account to the added zeros in sBar
[M,Ntilde] = size(B);
BBar = [B.' zeros(Ntilde,M-1)].';
% Gradient computation for frequency template error (FTE) metric (eq. 11)
gradF = 2./J.*(BBar.'*...
  imag(conj(sBar).*(ifft(ifftshift(((abs(sfBar).^2-u).*sfBar))))));
end

function alpha_k = backtracking_line_search(f,gradF,xk,pk)
% Implements backtracking line search using algorithm 3.1 in Nocedal (2006)
% Performing backtracking allows us to neglect the curvature Wolfe condition,
% which is fine for Newton methods but less fine for quasi-Newton and conjugate
% gradient methods.
%
% In pseudocode (from Nocedal p.37)
% Choose alphaBar > 0, rho E (0,1), c E (0,1), set alpha <- alphaBar
% repeat until f(x_k + alpha*p_k) <= f(x_k) + c*alpha*gradF_k^T*p_k
%     alpha <- rho*alpha;
% end(repeat)
% Terminate with alpha_k = alpha
% This translates directly into matlab
%
% Inputs:
%   - f: Function handle for objective function
%   - gradF: Function handle for objective function
%   - x: Current iterate
%   - pk: Current descent direction
%
% Outputs:
%   - alpha: The best candidate step length for the current iteration

% Initialize alpha
alphaBar = 1;
alpha = alphaBar;
% minAlpha = 0.1;
% Set the constants for the sufficient decrease condition
c = 0.1;
rho = 0.5;
while f(xk + alpha*pk) > f(xk) + c*alpha*gradF(xk).'*pk
  alpha = rho*alpha;
  if alpha < 10*eps
%     alpha = 0.1;
%     break
    error('Error in Line search - alpha close to working precision');
  end
  %   if alpha < minAlpha
  %     break
  %   end
end
alpha_k = alpha;
end
