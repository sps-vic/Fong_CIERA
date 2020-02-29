function [g_sdss, r_sdss, i_sdss, z_sdss] = PS1toSDSS(g_p1, r_p1, i_p1, z_p1);

%Determines SDSS magnitudes of stars from a list of PS1 g-band and r-band
%mags.

%From Equation 6 and Table 6 of Tonry+ 2012.

%Columns represent g, r, i and z coefficients
A0 = [0.013 -0.001 -0.005 0.013];
A1 = [0.145 0.004 0.011 -0.039];
A2 = [0.019 0.007 0.010 -0.012];

x = g_p1 - r_p1;

for i = 1:length(g_p1) %stepping through each star in the list
    g_SDSS = A0(1) + A1(1)*x(i) + A2(1)*x(i)^2 + g_p1;
    r_SDSS = A0(2) + A1(2)*x(i) + A2(2)*x(i)^2 + r_p1;
    i_SDSS = A0(3) + A1(3)*x(i) + A2(3)*x(i)^2 + i_p1;
    z_SDSS = A0(4) + A1(4)*x(i) + A2(4)*x(i)^2 + z_p1;
end

g_SDSS
r_SDSS
i_SDSS
z_SDSS