function [image] = grabImage()
    vid = videoinput('gentl', 1, 'Mono8');
    src = getselectedsource(vid);
    src.ExposureTime = 83716;
    vid.FramesPerTrigger = 1;
    %vid.ROIPosition = [236 355 1423 540];
    start(vid);
    image = getdata(vid);
end



