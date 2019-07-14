close all;
load('camParams.mat');
f2 = figure();

numberOfFeatures = 10;
windowSize = 5;

vid = videoinput('gentl', 1, 'Mono8');
src = getselectedsource(vid);
src.ExposureTime = 83716;
vid.FramesPerTrigger = 1;
%vid.ROIPosition = [236 355 1423 540];
%vid.ROIPosition = [724 456 484 304];
start(vid);




%SlidingWindow
features = zeros(numberOfFeatures, 2, windowSize); %(FeatureNr,XY,Time/Window) 
                                                    % Ad Each Step, the
                                                    % last Windows Contains
                                                    % the Positions of the
                                                    % Features from the
                                                    % Last Windows in here.
tracker = cell(windowSize,1);
for i = 1 : windowSize
   tracker{i} = vision.PointTracker();
end

viewID = 1;

image1 = imcrop(undistortImage(getdata(vid),cameraParams),[724 456 484 304]);;
corners = detectHarrisFeatures(image1);
corners = corners.selectStrongest(numberOfFeatures);
features(:,:,1) = corners.Location();    
for i = 1 : windowSize
   initialize(tracker{i},  features(:,:,1), image1); %fakeinit
end


%Norm
%x1 = x1 / size(image1,2);
%y1 = y1 / size(image1,1);



while true
    
    
    viewID = viewID + 1;
    image2 = imcrop(undistortImage(getdata(vid),cameraParams),[724 456 484 304]);
    
    %StepSlidingWindow--->
    for i = windowSize : -1 : 2
        tracker(i) = tracker(i - 1);
        features(:,:,i) = tracker{i}(image2);  
    end
    
    corners = detectHarrisFeatures(image2);
    corners = corners.selectStrongest(numberOfFeatures);
    features(:,:,1) = corners.Location();  
    tracker{1}(image2); 
    setPoints(tracker{1},  features(:,:,1));
    %<---
    
  
    
    
    figure(f2);clf;
    imshow(image2); hold on;
    plot(features(:,1,1),features(:,2,1),'x');
    plot(features(:,1,2),features(:,2,1),'o');
    plot(features(:,1,3),features(:,2,1),'*');
    plot(features(:,1,4),features(:,2,1),'.');
    plot(features(:,1,5),features(:,2,1),'+');
    
    %x2 = x2 / size(image2,2);
    %y2 = y2 / size(image2,1);
    
    %m2 = [x2,y2];
    
    %Unrotate
    %for i = 1 : length([x2,y2])
    %    m2(i,:) = hom2cart((rotation.' *  cart2hom(m2(i,:)).').');
    %end
    
    %m(:,:,1) = m2;
   
    b = calcBaseLine(features(:,:,2),features(:,:,1))
    %b = rotation.' * b;
    figure(10);
    vectarrow([0,0,0],[b(1), b(2), b(3)]);
    axis([-1 1 -1 1 -1 1]);
    
    %figure(11);
    %bkorr = groundTruthPoses.Location{imageID};
    %vectarrow([0,0,0],[bkorr(1), bkorr(2), bkorr(3)]);
    %axis([-1 1 -1 1 -1 1]);
    
    
end
