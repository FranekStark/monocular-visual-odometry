function costsum = costsum(m2,R2,x1,y1,z1,R1,m1,x0,y0,z0,R0,m0,a1,b1,t1,a0,b0,t0)
    costsum = 0;
    for i = 1:size(m0,2)
      costsum = costsum + cost(m2(:,i),R2,x1,y1,z1,R1,m1(:,i),x0,y0,z0,R0,m0(:,i),a1,b1,t1,a0,b0,t0);
    end
end

