function [b] = calcBaseLine(points1, points2)
    syms bx by bz;
    syms eqns;
    eqns = sym(zeros(1, length(points1)));
    for i = 1 : length(points1) 
        x1 = points1(i,1);
        y1 = points1(i,2);
        x2 = points2(i,1);
        y2 = points2(i,2);
        eqns(i) = -(y2 - y1)*bx + (x2-x1)*by + ((y2-y1)*x2-(x2-x1)*y2)*bz == 0;
    end
    [A,O] = equationsToMatrix(eqns, [bx, by, bz]);
    [U,S,V] = svd(A);
    b = V(:,end);% get last column of V
end

