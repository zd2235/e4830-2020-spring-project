clc,clear, close all
img = imread('p1-left/frames/0-eye.png');
img = im2double(rgb2gray(img));

% Step 1
radii = 60;% feature radius r

% response = 0;% response value
% op_C = zeros(size(img));% response heat map
% for r = 40:50
%     haar = haar_like_feature(r);
%     C = conv2(img,haar,'same');
%     [val,idx] = max(C(:));
%     if val>response
%         response = val;
%         [i_row, i_col] = ind2sub(size(C),idx);
%         radii = r;
%         op_C = C;
%     end
% end

% conv img with haar and find max response
haar = haar_like_feature(radii);
C = conv2(img,haar,'same');
[~,idx] = max(C(:));
[i_row, i_col] = ind2sub(size(C),idx);% max response index

% visualize haar_feature response heat map
figure();
subplot(121);imshow(haar);title('a.Haar-like feature');
subplot(122);imshow(img);title('b.Input data');
figure();imagesc(C);colormap('hot');title('c.response heat map');hold on;
rectangle('Position',[i_row-radii,i_col-radii,2*radii,2*radii],'LineWidth',2);
rectangle('Position',[i_row-3*radii,i_col-3*radii,6*radii,6*radii],'LineWidth',2);
fprintf('haar feature radius: %d\n',radii);

% crop pupil region
pupil = imcrop(img,[i_row-radii,i_col-radii,2*radii,2*radii]);

% Binarize tresholding via k-means
label = kmeans(pupil(:),2);
label = reshape(label,size(pupil));
mask = double(label>1);
seg_pupil = mask.*label;
% fix pupil region
se = strel('disk',9);
op_pupil = imclose(seg_pupil,se);
% find largest connected component & pupil center
CC = bwconncomp(op_pupil);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx1] = max(numPixels);
mass = zeros(size(seg_pupil));
mass(CC.PixelIdxList{idx1})=1;
center = regionprops(mass, 'Centroid');

figure();
subplot(121);imshow(pupil);title('a.cropped pupil region');
subplot(122);imshow(seg_pupil);title('c.segmented pupil');
figure();histogram(pupil);title('b.intensity histogram');

figure();
subplot(121);imshow(op_pupil);title('a.Fixed pupil with center');
h = drawpoint('Position',center.Centroid); h.Label = 'center';

% Edge detection
% se = strel('disk',9);
% open = imopen(pupil,se);
bw = edge(op_pupil,'Canny',[0.3,0.5]);
subplot(122);imshow(bw);title('b.Canny edge');
% figure();
% subplot(131);imshow(pupil);title('pupil region');
% subplot(132);imshow(open);title('morphological open');
% subplot(133);imshow(bw);title('canny edges');

% extract edge coordinates for ellipse fitting
[m,n] = find(bw);
coordinate = cat(2,n+i_row-radii,m+i_col-radii);
figure();
imshow(img);hold on;
plot(coordinate(:,1),coordinate(:,2),'r');hold off;

% rand_pick = randperm(length(coordinate),5);% pick 5 random points
% rand_points = coordinate(rand_pick,:);
% plot(rand_points(:,1),rand_points(:,2),'*'); hold off;
% [Gx,Gy] = imgradientxy(bw);
% rejection = 0;
% 
% while(1) % RANSAC iteration
%     rand_pick = randperm(length(coordinate),5);% pick 5 random points
%     rand_points = coordinate(rand_pick,:);
%     Q = EllipseDirectFit(rand_points);
% 
%     for i=1:5 % early sample rejection
%         t = 1;
%         Q_x = 2*Q(1)*rand_points(i,1)+Q(2)+Q(4);
%         Q_y = 2*Q(3)*rand_points(i,2)+Q(2)+Q(5);
%         grad = Q_x*Gx(rand_points(i,:)) + Q_y*Gy(rand_points(i,:));
%         if grad <= t
%             t = grad;
%         end      
%     end
%     if grad <= 0
%         break
%     end
% end
% plot(rand_points(:,1),rand_points(:,2),'*');
% % syms x y
% % fimplicit(Q(1)*x^2 + Q(2)*x*y + Q(3)*y^2 + Q(4)*x + Q(5)*y +Q(6)==1);

% function to build haar feature
function haar = haar_like_feature(r)
haar = ones(6*r+1,6*r+1);
Nin = (2*r+1)^2;
Nout = (6*r+1)^2-Nin;
haar(2*r+1:4*r+1,2*r+1:4*r+1) = -Nout/Nin;
end


% function A = EllipseDirectFit(XY)
% %
% %  Direct ellipse fit, proposed in article
% %    A. W. Fitzgibbon, M. Pilu, R. B. Fisher
% %     "Direct Least Squares Fitting of Ellipses"
% %     IEEE Trans. PAMI, Vol. 21, pages 476-480 (1999)
% %
% %  Our code is based on a numerically stable version
% %  of this fit published by R. Halir and J. Flusser
% %
% %     Input:  XY(n,2) is the array of coordinates of n points x(i)=XY(i,1), y(i)=XY(i,2)
% %
% %     Output: A = [a b c d e f]' is the vector of algebraic 
% %             parameters of the fitting ellipse:
% %             ax^2 + bxy + cy^2 +dx + ey + f = 0
% %             the vector A is normed, so that ||A||=1
% %
% %  This is a fast non-iterative ellipse fit.
% %
% %  It returns ellipses only, even if points are
% %  better approximated by a hyperbola.
% %  It is somewhat biased toward smaller ellipses.
% %
% centroid = mean(XY);   % the centroid of the data set
% D1 = [(XY(:,1)-centroid(1)).^2, (XY(:,1)-centroid(1)).*(XY(:,2)-centroid(2)),...
%       (XY(:,2)-centroid(2)).^2];
% D2 = [XY(:,1)-centroid(1), XY(:,2)-centroid(2), ones(size(XY,1),1)];
% S1 = D1'*D1;
% S2 = D1'*D2;
% S3 = D2'*D2;
% T = -inv(S3)*S2';
% M = S1 + S2*T;
% M = [M(3,:)./2; -M(2,:); M(1,:)./2];
% [evec,~] = eig(M);
% cond = 4*evec(1,:).*evec(3,:)-evec(2,:).^2;
% A1 = evec(:,find(cond>0));
% A = [A1; T*A1];
% A4 = A(4)-2*A(1)*centroid(1)-A(2)*centroid(2);
% A5 = A(5)-2*A(3)*centroid(2)-A(2)*centroid(1);
% A6 = A(6)+A(1)*centroid(1)^2+A(3)*centroid(2)^2+...
%      A(2)*centroid(1)*centroid(2)-A(4)*centroid(1)-A(5)*centroid(2);
% A(4) = A4;  A(5) = A5;  A(6) = A6;
% A = A/norm(A);
% end  %  EllipseDirectFit