syms w1 w2 w3 real
w_hat = [0 -w3 w2;
    w3 0 -w1;
    -w2 w1 0];
disp(w_hat^3);