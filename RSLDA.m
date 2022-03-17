function  [W, ratiosumsave]  = RSLDA( d,m, S_w, S_b, sigma,maxiterout,maxiterinner)

% Solve the Ratio Sum problems: 
% max_{W'*W=I}   SUM[(w_k'*A*w_k)/(w_k'*B*w_k)
% The matrix A should be symmetric and B must be positive semi-definite.
% The cross set of null spaces of A and B should be null.

% Author£ºHongmei Wang , Northwestern Polytical University.
% 2020-08
% ref: 
% "Ratio Sum vs. Sum Ratio for Linear Discriminant Analysis," 
% in IEEE Transactions on Pattern Analysis and Machine Intelligence

% S_w: Within-class divergence matrix
% S_b: Between-class divergence matrix
% d: Original data dimension
% m: Matrix dimension after projection
% W: Projection matrix
% sigma: Adjustable parameters
% epsilon: Convergence accuracy

% Initialize the projection matrix and times

t = 1;
I_d = eye(d);
I_m =eye(m);
SSW=sigma*I_d-S_w;
C0=rand(d,m);
[U,S,V] = svd(C0);
W0 = U*[I_m;zeros(d-m,m)]*V';
% W0=orth(rand(d,m));
Cmatrix = [];
ratiosum=0;

while t<=maxiterout
    % Update parameter
    for k = 1:m
        w_k = W0(:,k);
        % Update lambda_k
        lambda1=sqrt((w_k')*S_b*w_k);
        lambda2=((w_k')*S_w*w_k);
        lambda(1,k) = lambda1/lambda2;
        % Update c_k
    end
    W1=W0;
    t0=0;
    while  t0<=maxiterinner
        W2=W1;
        %    while  ew>epsilon
        Cmatrix=zeros(d,m);
        for k = 1:m
            w1_k = W1(:,k);
            Cmatrix(:,k) = (2*lambda(1,k)*lambda(1,k)*SSW+2*lambda(1,k)*S_b/sqrt((w1_k')*S_b*w1_k))*w1_k;
        end
        % Singular value decomposition C
        [U,S,V] = svd(Cmatrix);
        % Update W
        W1 = U*[I_m;zeros(d-m,m)]*V';
        t0=t0+1;
    end
    % Calculate ratio sum value
%     ratiosum = 0;
%     for k = 1:m
%         w2_k = W1(:,k);
%         %ratiosum = ratiosum+(((w2_k')*S_b*w2_k)/((w2_k')*S_w*w2_k));
%         ratiosum =ratiosum+(lambda(1,k)*lambda(1,k)*((w2_k')*S_w*w2_k)-2*lambda(1,k)*sqrt((w2_k')*S_b*w2_k));
%     end
%     ratiosumsave(t) = ratiosum;
    W0=[];
    W0=W1;
    t=t+1; 
end
% e=ratiosumsave(2:end)-ratiosumsave(1:end-1);

W=W0;

