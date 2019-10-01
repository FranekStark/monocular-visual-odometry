figure(1);
title('Vec 0');
vectarrow([0;0;0],[x0;y0;z0], "estimate");
%axis([0 2 0 2 0 2]);
hold on;
figure(2);
title('Vec 1');
vectarrow([0;0;0],[x1;y1;z1], "estimate");
%axis([0 2 0 2 0 2]);
hold on;

for i = 1:4:40
    params = solve([0;0;1;0;0;1],i * 200*6,i * 400,m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0);
    a0r = params(1);
    b0r = params(2);
    t0r = params(2);
    a1r = params(1);
    b1r = params(2);
    t1r = params(2);
    vec0 = scale(t0r) * baseLine(x0,y0,z0,a0r,b0r);
    vec1 = scale(t1r) * baseLine(x1,y1,z1,a1r,b1r);
    figure(1);
    vectarrow([0;0;0], vec0, num2str(i));
    hold on;
    figure(2);
    vectarrow([0;0;0], vec1,num2str(i));
    hold on;
end