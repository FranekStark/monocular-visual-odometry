function [x,fval,exitflag,output,jacobian] = solve(firstParams,MaxFunctionEvaluations_Data,MaxIterations_Data,m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0)
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options
options = optimoptions('fsolve');
%% Modify options setting
%options = optimoptions(options,'Display', 'iter-detailed');
options = optimoptions(options,'MaxFunctionEvaluations', MaxFunctionEvaluations_Data);
options = optimoptions(options,'MaxIterations', MaxIterations_Data);
options = optimoptions(options,'Algorithm', 'levenberg-marquardt');
options = optimoptions(options,'FiniteDifferenceType', 'central');
[x,fval,exitflag,output,jacobian] = ...
fsolve(@(params)costsum2(m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0,params),firstParams,options);
