function [yte,ype,yp_te,ytr,ypr,yp_tr,RMSE_tr,RMSE_te,task_id,Avetime] = UnsuperFea_ELLA(x_tr,y_tr,x_te,y_te,k,rho,rho2,rou,T_pool,taskSelec,zero_shot)
%% ini
[~,T_max]=size(x_tr);
[~,d]=size(x_tr{1});
me_num1=d+1;
me_num2=d;

A1=zeros(k*me_num1,k*me_num1);
b1=zeros(k*me_num1,1);
A2=zeros(k*me_num2,k*me_num2);
b2=zeros(k*me_num2,1);
L=0.5*ones(me_num1,k);
D=0.5*ones(me_num2,k);
%L=randn(me_num1,k);
%D=randn(me_num2,k);

opts.max_iter=10^6;
opts.tol=10^(-10);
opts.tFlag=0;
opts=init_opts(opts);

T=0;
e=0;
re=10^-5*diag([0, ones(1,d)]); % ridg term
task_id=[]; 

%% algorithm

start_time=tic;
for t=1:T_pool
  %% task selection   
  if taskSelec==1 | taskSelec == 2

 iter=0;
 D_pool=[];theta_pool=[];s_pool=[];cost=[];
 for iter=1:T_pool
     if ismember(iter,task_id)==0
     X=x_tr{iter}';
     Y=y_tr{iter};
     X_new=[ones(1,size(X,2));X]; 
     % model
     theta_pool=pinv(X_new*X_new'+re )*X_new*Y; 
     D_pool=(X_new*X_new'+re)/(2*size(X_new,2));
     % unsuper fea
     %{
     [theta_pool2] = unsuper_FeaCoding(X);
     D_pool2=rou*eye(me_num2);
     % combine
     K=[L;D];
    betal_pool=[theta_pool;theta_pool2];
     A_pool=[D_pool,zeros(me_num1,me_num2);zeros(me_num2,me_num1),D_pool2];
     % solve s
     [s_pool, ~] = my_Lasso(K, betal_pool, A_pool, rho, opts);
     cost(iter)=rho*abs(s_pool)+(betal_pool-K*s_pool)'*A_pool*(betal_pool-K*s_pool);
   %}
     [s_pool, ~] = my_Lasso(L, theta_pool, D_pool, rho, opts);
     cost(iter)=(theta_pool-L*s_pool)'*D_pool*(theta_pool-L*s_pool);
     else
     cost(iter)=0;
     end
 end
 
if taskSelec==1 % diversity
[cost_max(t),idx]=max(cost);
task_id(t)=idx;
%end

 else % diversity++
probs = cost.^2./sum(cost.^2);

     
 end

 if length(task_id)-length(unique(task_id))~=0 % check task order is correct
    error('stupid!');
 end
 
elseif taskSelec==0  % random task 
 task_id(t)=t;
  end
  
 %% knowledge base update
    T=T+1;
    X=x_tr{task_id(t)}';
    Y=y_tr{task_id(t)};

% model 
X_new=[ones(1,size(X,2));X]; 
theta1{t}=pinv(X_new*X_new'+re )*X_new*Y; 
D1{t}=(X_new*X_new'+re)/(2*size(X_new,2));
%R=sqrtm(D1{t});


[theta2{t}] = unsuper_FeaCoding(X);
D2{t}=rou*eye(me_num2);

% compute s coupled 
betal{t}=[theta1{t};theta2{t}];
A=[D1{t},zeros(me_num1,me_num2);zeros(me_num2,me_num1),D2{t}];
K=[L;D];
[s{t}, funcVal] = my_Lasso(K, betal{t}, A, rho, opts);
   
% update L
A1=A1+kron( (s{t}*s{t}')',D1{t});
b1=b1+ ( kron( s{t}', theta1{t}'*D1{t} ) )';
L_vec=inv((1/T)*A1+rho2*eye(k*me_num1,k*me_num1))*(1/T)*b1;
L=reshape(L_vec,[me_num1,k]);

% update D
A2=A2+kron( (s{t}*s{t}')',D2{t});
b2=b2+ ( kron( s{t}', theta2{t}'*D2{t} ) )';
D_vec=inv((1/T)*A2+rho2*eye(k*me_num2,k*me_num2))*(1/T)*b2;
D=reshape(D_vec,[me_num2,k]);

%% evaluation
time2=tic;
if t==T_pool % learn all tasks
%% training task evaluation

s=cell2mat(s);
theta_tr=L*s; 

for i=1:T_pool
    yp_tr{i}=[ones(size(x_te{i},1),1),x_te{i}]*theta_tr(:,i);
    
task_tr_idx(i)=size(yp_tr{i},1);
task_tr_idxsum(i)=sum(task_tr_idx(1:i));
    if i==1
        a=1;
    else
        a=task_tr_idxsum(i-1)+1;
    end
    ytr(a:task_tr_idxsum(i))=y_te{i}; 
    ypr(a:task_tr_idxsum(i))=yp_tr{i}; 
end
RMSE_tr = sqrt(mean((ytr - ypr).^2));
end
%% testing task evaluation 
count=0;
for j=T_pool+1:T_max
 count=count+1;   
X=x_tr{j}';
Y=y_tr{j};
X_new=[ones(1,size(X,2));X]; 
theta_es1=pinv(X_new*X_new'+re )*X_new*Y; 

D_eva1=(X_new*X_new'+re)/(2*size(X_new,2));
[theta_es2] = unsuper_FeaCoding(X);
D_eva2=rou*eye(me_num2);
K=[L;D];
betal_es=[theta_es1;theta_es2];
A_eva=[D_eva1,zeros(me_num1,me_num2);zeros(me_num2,me_num1),D_eva2];


if zero_shot==0
[s_eva{j}, ~] = my_Lasso(K, betal_es, A_eva, rho, opts);
elseif zero_shot==1 % unsupervised transfer
[s_eva{j}, ~] = my_Lasso(D, theta_es2, D_eva2, rho, opts);
else % model transfer
[s_eva{j}, ~] = my_Lasso(L, theta_es1, D_eva1, rho, opts);
end

theta_eva{j}=L*s_eva{j};
yp_te{j}=[ones(size(x_te{j},1),1),x_te{j}]*theta_eva{j};

%% 
task_te_idx(count)=size(yp_te{j},1);
task_te_idxsum(count)=sum(task_te_idx(1:count));
    if count==1
        a=1;
    else
        a=task_te_idxsum(count-1)+1;
    end
    yte(a:task_te_idxsum(count))=y_te{j}; 
    ype(a:task_te_idxsum(count))=yp_te{j}; 
    
   %{ 
   % if nor_method~=1
   % yte1(a:task_te_idxsum(count))=mapminmax('reverse',y_te{j},PS{j}); % Denormalization
   % ype1(a:task_te_idxsum(count))=mapminmax('reverse',yp_te{j},PS{j}); % Denormalization
   % end
   %} 
end
time_eva(t)=toc(time2);
RMSE_te1(t) = sqrt((mean((yte - ype).^2)));


end
RMSE_te=RMSE_te1;
Avetime=(toc(start_time)-sum(time_eva))*10^3/T_pool;

end