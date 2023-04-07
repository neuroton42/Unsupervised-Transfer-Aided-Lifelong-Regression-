function [W, funcVal] = my_Lasso(X, Y, D, rho1, opts)

if nargin <3
    error('\n Inputs: X, Y, abd rho1 should be specified!\n');
end
% cell X array transpose ;every cell is n x d; when tansposed:
% row:feature number (d)
%column:sample number (n)
%X = multi_transpose(X);
X=X';

if nargin <4
    opts = [];
end

% initialize options.
opts=init_opts(opts);

if isfield(opts, 'rho_L2')
    rho_L2 = opts.rho_L2;
else
    rho_L2 = 0;
end

% X contains many matrix(n x d)?the number of task number is length(X)
%task_num  = length (X);
task_num  =1; %%%%%%%% only one task each time

%every cell's demension is d
dimension = size(X, 1);
%store function value
funcVal = [];

%create cell to store result in every task
%XY = cell(task_num, 1);
XY=[];
%
W0_prep = [];

% initialize a weight
%every XY{t_idx} is the result of every task(dxn  x   nx1)
%put the result of every task(dxn  x   nx1) into W0_prep
for t_idx = 1: task_num
    XY = X*Y;
    W0_prep = cat(2, W0_prep, XY);
end

% initialize a starting point
if opts.init==2
    W0 = zeros(dimension, task_num);
elseif opts.init == 0
    W0 = W0_prep;
else
    if isfield(opts,'W0')
        W0=opts.W0;
        if (nnz(size(W0)-[dimension, task_num]))
            error('\n Check the input .W0');
        end
    else
        W0=W0_prep;
    end
end


bFlag=0; % this flag tests whether the gradient step only changes a little

Wz= W0;
Wz_old = W0;

t = 1;
t_old = 0;


iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    %regularizaion parameter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    
    % compute function value and gradients of the search point
    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval  (Ws);
    
    while true
        
        [Wzp, l1c_wzp] = l1_projection(Ws - gWs/gamma,   rho1 / gamma);
%       [Wzp, l1c_wzp] = l1_projection(Ws - gWs/gamma,  2 * rho1 / gamma);
        
        Fzp = funVal_eval  (Wzp);
        
        delta_Wzp = Wzp - Ws;
        r_sum = norm(delta_Wzp, 'fro')^2;
        Fzp_gamma = Fs + sum(sum(delta_Wzp .* gWs)) + gamma/2 * sum(sum(delta_Wzp.*delta_Wzp));
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Wz = Wzp;
    
    funcVal = cat(1, funcVal, Fzp + rho1 * l1c_wzp);
    
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

W = Wzp;

% private functions

    function [z, l1_comp_val] = l1_projection (v, beta)
        % this projection calculates
        % argmin_z = \|z-v\|_2^2 + beta \|z\|_1
        % z: solution
        % l1_comp_val: value of l1 component (\|z\|_1)
        z = sign(v).*max(0,abs(v)- beta/2);
         
%         l1_comp_val = sum(sum(abs(z)));
        l1_comp_val = sum(abs(z));
        %l1_comp_val = max(sum(abs(z)));
    end

    function [grad_W] = gradVal_eval(W)
        grad_W = [];
        for t_ii = 1:task_num
            %{
            XWi = X' * W;
            XTXWi = X* XWi;
            grad_W = cat(2, grad_W, XTXWi - XY);
            %}   
   grad_W = cat(1, grad_W, -2 * X*D*Y + 2 * X*D*X'*W);  % Menghui Zhou           
        end
       % grad_W = grad_W + rho_L2 * 2 * W;
    end

    function [funcVal] = funVal_eval (W)
        funcVal = 0;
        for i = 1: task_num
           funcVal = funcVal + (Y - X' * W)'*D*(Y - X' * W); 
        end
        %funcVal = funcVal + rho_L2 * norm(W, 'fro')^2;
    end


end