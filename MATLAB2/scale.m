function [sc] = scale(t)
    LOW = 0.25;
    HIGH = 2;
  
    sc = LOW + (HIGH - LOW) / (1 + exp(-t));
end

