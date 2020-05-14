clc,clear,close all
xCenter = 194.015831;yCenter = 203.153840; a=46.952739;b=22.625909;alpha=-0.529757;
theta = 0 : 0.01 : 2*pi;
x = a * cos(theta);
y = b * sin(theta);
R  = [cos(alpha) -sin(alpha); ...
      sin(alpha)  cos(alpha)];
rCoords = R*cat(1,x,y);
xr = rCoords(1,:)' + xCenter;      
yr = rCoords(2,:)' + yCenter;

img = imread('p1-left/frames/0-eye.png');
img = im2double(rgb2gray(img));

i=imshow(img);title('a.label ellipse');hold on;
plot(xr,yr,'r');hold off;