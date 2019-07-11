function [points] = readPoints(number)
    points = zeros(number,2);
    for i = 1 : number
        point = drawpoint;
        point.Label = num2str(i);
        points(i,1) = point.Position(1);
        points(i,2) = point.Position(2);
    end
end

