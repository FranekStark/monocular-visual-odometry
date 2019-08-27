function c = cost(mk2, Rk2, x1, y1, z1, Rk1, mk1, x,y,z,Rk,mk,a1,b1,t1,a,b,t)
    LOW = 0.25;
    HIGH = 2;
    
    nk = LOW + (HIGH - LOW) / (1 + exp(t));
    nk1 = LOW + (HIGH - LOW) / (1 + exp(t1));

    uk = [(1-(a^2))  (-2*a) (0); (2*a)  (1-(a^2))  (0); (0) (0) (1)] * [(1-(b^2))  (0)  (2*b); (0)  (1) (0); (-2*b)  (0) (1-(b^2))] / (1+b^2)/(1+a^2)  * [x;y;z];
    uk1 = [(1-(a1^2))  (-2*a1) (0); (2*a1)  (1-(a1^2))  (0); (0) (0) (1)] * [(1-(b1^2))  (0)  (2*b1); (0)  (1) (0); (-2*b1)  (0) (1-(b1^2))] / (1+b1^2)/(1+a1^2)  * [x1;y1;z1];   
    
    c = power(mk2' * Rk2' * cross(uk1, Rk1 * mk1),2) + ...
        power(mk1' * Rk2' * cross(uk, Rk * mk),2) + ...
        power(mk2' * Rk2' * cross((nk1*uk1+nk*uk)/norm(nk1*uk1+nk*uk), Rk * mk),2);
end