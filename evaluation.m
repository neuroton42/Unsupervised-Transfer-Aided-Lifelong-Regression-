clear all;
%% school dataset 
nor_method=1; % 1: whole task normalization /
data=struct2cell(importdata('school.mat'));
x_cell=data(1,:); x_cell=x_cell{1,1};
y_cell=data(2,:); y_cell=y_cell{1,1};
T_max=size(y_cell,2);

for i=1:T_max
    task_idx(i)=size(x_cell{i},1);
    task_idxsum(i)=sum(task_idx(1:i));
    if i==1
        b=1;
    else
        b=task_idxsum(i-1)+1;
    end
    x_raw(b:task_idxsum(i),:)=x_cell{i}; 
    y_raw(b:task_idxsum(i),:)=y_cell{i};   
end

x=x_raw';
y=y_raw';

if nor_method==1
for i=1:size(x,1)
[x(i,:),~]=mapminmax(x(i,:),0,1); % x:d*N y:1*N
end
[y,PS]=mapminmax(y,0,1);
end
%% train/test

K=T_max;
x=x'; % x:N*d
% task
for i=1:K
    if i==1
    xc1{i}=x(1:task_idx(i),:);   
    yc1{i}=y(1:task_idx(i));
    else
    a=sum(task_idx(1:i-1));
    xc1{i}=x(a+1:a+task_idx(i),:);
    yc1{i}=y(a+1:a+task_idx(i));
    end

if nor_method~=1
for j=1:size(xc1{i},2)
[xc1{i}(:,j),~]=mapminmax(xc1{i}(:,j)',0,1); % x:d*N y:1*N
end
[yc1{i},PS{i}]=mapminmax(yc1{i},0,1);
end

end

iter=1;
% training /testing
training_percent=0.5;
task_num{iter} = randperm(K);  
for i=1:K    
    % rand task sequence 
    xc{i}=xc1{task_num{iter}(i)};
    yc{i}=yc1{task_num{iter}(i)};
    
    x_tr{i}=xc{i}(1:round( size(xc{i},1)*training_percent ),:);
    y_tr{i}=yc{i}(1:round( size(xc{i},1)*training_percent ))';
    x_te{i}=xc{i}(round( size(xc{i},1)*training_percent )+1:end,:);
    y_te{i}=yc{i}(round( size(xc{i},1)*training_percent )+1:end)';
    %}
end
T_max=length(task_idx);
%% parameter
k=1; %                  
rho=10^-5;  %            
rho2=10^-5; %    
T_pool=round(T_max/2);
taskSelec=0; % 0: randon/ 1: diversity / 2: diversity++
zero_shot=0; % 0: both / 1: unsuper_fea / 2: model
rou=0.1; % shcool:0.1 \pd: 0.01 /ad:5

%% algorithm
[yte{iter},ype{iter},yp_te,ytr{iter},ypr{iter},yp_tr,RMSE_tr1(iter),RMSE_te1(iter,:),task_id,ACTpT(iter)] =...
    UnsuperFea_ELLA(x_tr,y_tr,x_te,y_te,k,rho,rho2,rou,T_pool,taskSelec,zero_shot);
%% results
RMSE_tr(iter) = sqrt(mean((ytr{iter} - ypr{iter}).^2));
RMSE_te(iter) = sqrt((mean((yte{iter} - ype{iter}).^2)));




