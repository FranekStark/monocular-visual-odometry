function c = cost(mk2, Rk2, x1, y1, z1, Rk1, mk1, x,y,z,Rk,mk,a1,b1,t1,a,b,t)
   
    nk = scale(t);
    nk1 = scale(t1);

    uk = baseLine(x,y,z,a,b);
    uk1 = baseLine(x1,y1,z1,a1,b1);
    
    c = power(mk2' * Rk2' * cross(uk1, Rk1 * mk1),2) + ...
        power(mk1' * Rk2' * cross(uk, Rk * mk),2) + ...
        power(mk2' * Rk2' * cross((nk1*uk1+nk*uk)/norm(nk1*uk1+nk*uk), Rk * mk),2);
end