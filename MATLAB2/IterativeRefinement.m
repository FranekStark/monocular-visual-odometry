
x = -1:0.0001:1;

f = @(a0) costsum(m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0,a1,b1,t1,a0,b0,t0);

y = arrayfun(f,x);



plot(x,y);

%%
x = 0:pi/100:2*pi;
y = sin(x);
plot(x,y)