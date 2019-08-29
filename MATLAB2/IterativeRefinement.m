clear;
run dataTEMP.m;

%%
x = -2:0.01:2;


fa1 = @(a1) costsum(m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0,a1,b1,t1,a0,b0,t0);
fb1 = @(b1) costsum(m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0,a1,b1,t1,a0,b0,t0);
fa0 = @(a0) costsum(m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0,a1,b1,t1,a0,b0,t0);
fb0 = @(b0) costsum(m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0,a1,b1,t1,a0,b0,t0);
ft1 = @(t1) costsum(m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0,a1,b1,t1,a0,b0,t0);
ft0 = @(t0) costsum(m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0,a1,b1,t1,a0,b0,t0);




ya1 = arrayfun(fa1,x);
yb1 = arrayfun(fb1,x);
ya0 = arrayfun(fa0,x);
yb0 = arrayfun(fb0,x);
yt1 = arrayfun(ft1,x);
yt0 = arrayfun(ft0,x);

plot(x,ya1,x,yb1,x,ya0,x,yb0,x,yt1,x,yt0);
legend({'a1','b1','a0','b0','t1','t0'},'Location','southwest');

fall = @(params) costsum2(m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0,params);


%%

f1= @(x,y) costsum(m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0,x,y,t1,a0,b0,t0);
f0= @(x,y) costsum(m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0,a1,b1,t1,x,y,t0);
figure(2);
fsurf(f1);
figure(3);
fsurf(f0);
%%

