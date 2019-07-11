close all;
iteration = 0;
loop = true;
load('camParams.mat')
im2 = undistortImage(grabImage(), cameraParams);
position =[0;0;0];
figure(20)
grid on;
plotx = animatedline('Color', 'red');
ploty = animatedline('Color', 'green');
plotz = animatedline('Color', 'yellow');
addpoints(plotx, iteration, 0);
drawnow;
addpoints(ploty, iteration, 0);
drawnow;
addpoints(plotz, iteration, 0);
drawnow;
while loop % loop of infinity
iteration=iteration+1;
%%Images
%subplot(1,2,1);
%im1 = imread('back.JPEG');A 
%pic1 = imshow(im1);
%points1 = readPoints(4);
%%
%pause;
%subplot(1,2,2);
%im2 = imread('back.JPEG');
%pic2 = imshow(im2);
%points2 = readPoints(4);
%pic2 = imshow(im2);
%points2 = readPoints(4);
pause();
im1 = im2;
im2 = undistortImage(grabImage(), cameraParams);

figure(1)
subplot(1,2,1); 
imshow(im1);
%points1 = detectSURFFeatures(im1);
%plot(points1.selectStrongest(10));
points1 = readPoints(4);
%im2 = undistortImage(grabImage(), cameraParams);
%im2 = im1;
subplot(1,2,2);
imshow(im2);
%points2 = detectSURFFeatures(im2);
%plot(points2.selectStrongest(10));
points2 = readPoints(4);


%% Extract Features
%[features1,  validPts1]  = extractFeatures(im1,  points1);
%[features2, validPts2] = extractFeatures(im2, points2);

%indexPairs = matchFeatures(features1, features2);

%matched1  = validPts1(indexPairs(:,1));
%matched2 = validPts2(indexPairs(:,2));

%figure(2);
%showMatchedFeatures(im1,im2,matched1,matched2);



%% Create Lineare Equation
syms bx by bz
syms eqns
for i = 1 : 4 %matched1.Count
    x1 = points1(i,1);
    y1 = points1(i,2);
    x2 = points2(i,1);
    y2 = points2(i,2);
   %x1 = matched1.Location(i,1);
   x1 = x1 / size(im1,2);
   %y1 = matched1.Location(i,2);
   y1 = y1 / size(im1,1);
   %x2 = matched2.Location(i,1);
   x2 = x2 / size(im2,2);
   %y2 = matched2.Location(i,2);
   y2 = y2 / size(im2,1);
   eqns(i) = -(y2-y1)*bx + (x2-x1)*by + ((y2-y1)*x2-(x2-x1)*y2)*bz == 0;
end


%% Linear Equation A * b = 0
%syms bx by bz
%eqn = -(y2-y1)*bx + (x2-x1)*by + ((y2-y1)*x2-(x2-x1)*y2)*bz == 0;
[A,O] = equationsToMatrix(eqns, [bx, by, bz]);

%%
%b = linsolve(A,O)
%c = A\O
%d = inv(transpose(A) * A) * transpose(A) * O
[U,S,V] = svd(A);
e = V(:,end)% get last column of V
%%
position = position + e
figure(10);
vectarrow([0,0,0],[e(1), e(2), e(3)]);
%addpoints(plotline, double(position(1)), double(position(2)), double(position(3)));

figure(20)
addpoints(plotx, iteration, double(position(1)));
drawnow;
addpoints(ploty, iteration, double(position(2)));
drawnow;
addpoints(plotz, iteration, double(position(3)));
drawnow;
hold on;
end
