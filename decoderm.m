function xhat=decoderm(r,alpha,puncture,L_c,L_total,dec_alg,niter,g,m)
yk = demultiplex(r,alpha,puncture); 
% demultiplex to get input for decoder 1 and 2
     
% Scale the received bits    
rec_s = 0.5*L_c*yk;
% Initialize extrinsic information      
L_e(1:L_total) = zeros(1,L_total);

for iter = 1:niter
    % Decoder one
    L_a(alpha) = L_e;  % a priori info. 
    if strcmp(dec_alg,"logmap")
         L_all = logmapo(rec_s(1,:), g, L_a, 1);  % complete info.
    else   
         L_all = sova0(rec_s(1,:), g, L_a, 1);  % complete info.
    end   
    L_e = L_all - 2*rec_s(1,1:2:2*L_total) - L_a;  % extrinsic info.

    % Decoder two         
    L_a = L_e(alpha);  % a priori info.
    if strcmp(dec_alg,"logmap")
         L_all = logmapo(rec_s(2,:), g, L_a, 2);  % complete info.  
    else
         L_all = sova0(rec_s(2,:), g, L_a, 2);  % complete info. 
    end
         L_e = L_all - 2*rec_s(2,1:2:2*L_total) - L_a;  % extrinsic info.   
    % Estimate the info. bits        
    xhat(alpha) = (sign(L_all)+1)/2;
end	%iter
xhat=xhat(1:L_total-m);
xhat=xhat';