clc
clear all

% import data
x_data = [0.00 0.24 0.48 0.72 0.95]';
y_data = [1.15 0.55 -1.12 -1.04 0.97]';

% produce coefficients of model eqaution 
a = myFit(x_data, y_data, @yFunctions); 

% produce extrapolated values
x_extrapolate = linspace(min(x_data), 1.1*max(x_data), 200)';
y_extrapolate = yFunctions(x_extrapolate)*a;

plot(x_data, y_data, "or", x_extrapolate, y_extrapolate, 'b', x_extrapolate, y_extra2, 'r')
grid on; grid minor
a
% Determine accuracy of model
R2 = accuracy(x_data, y_data, a);


%% Functions

% function to determine a
function a = myFit(x, y, f)
    a = f(x)\y;
end 

% function to fit graph
function y = yFunctions(x)
    n = length(x);
    y = [sin(2*pi*x), cos(2*pi*x)]; 
end

% function to determine accuracy
function R2 = accuracy(x_data, y_data, a_coefficients)
    y_fit = yFunctions(x_data) * a_coefficients;
    R = corrcoef(y_data, y_fit);
    R2 = R(2,1)^2;
end