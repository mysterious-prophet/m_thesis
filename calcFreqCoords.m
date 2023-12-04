%% Calculate 3D frequency coordinates
% calculates 3D spherical frequency coordinates k, mu, psi based on 3D 
% image size M, N, P

% inputs: M, N, P - size of 3D image
% outputs: omega - 3D meshgrid
%        : k - modulus
%        : mu - azimuth angle -pi < mu <= pi, phi equivalent
%        : psi - polar angle 0 <= psi <= pi, theta equivalent

function [k, mu, psi] = calcFreqCoords(M, N, P)
    om_1 = ((0:M-1) ./ M) * 2*pi - pi;
    om_2 = ((0:N-1) ./ N) * 2*pi - pi;
    om_3 = ((0:P-1) ./ P) *pi - pi/2;

    [omega_1, omega_2, omega_3] = meshgrid(om_1, om_2, om_3);

    k = sqrt(omega_1.^2 + omega_2.^2 + omega_3.^2);
    mu = angle(omega_1 + 1i*omega_2);
    psi = acos(omega_3 ./ k);
end