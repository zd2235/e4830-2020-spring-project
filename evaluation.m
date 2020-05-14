clc,clear,close all

label = imread('p1-left/0-label.png');
test = imread('p1-left/0-test.png');

% extract red region (pupil)
[Rl,Gl,Bl] = imsplit(label);
maskl = 255*uint8(Rl == 255 & Gl == 0 & Bl == 0);

[Rt,Gt,Bt] = imsplit(test);
maskt = 255*uint8(Rt == 255 & Gt == 0 & Bt == 0);

% set calculation
lt = (maskl + maskt)/255;
MO = (maskl/255 + maskt/255)-lt;

% evaluate misclassified area
L = imbinarize(maskl);
lt = imbinarize(lt); % label union test
MO = imbinarize(MO); % label intersect test
L = regionprops(L,'Area');
I = regionprops(MO,'Area');
U = regionprops(lt,'Area');

error = 2*(L.Area-I.Area) / U.Area;
fprintf('Misclassified area ratio is %f\n',error);