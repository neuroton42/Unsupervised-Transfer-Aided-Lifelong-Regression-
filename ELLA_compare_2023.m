clear all;
%clc;

dataset=4;
nor_method=1; % 1: whole task normalization /
%%
if dataset==1
%% school data

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
%}
elseif dataset==2
%% PD 1

A=importdata('park.txt');
%te=570; % 570 1180 1970
sub=(A(1:end,1)); % 1 index
data=(A(1:end,[5,7:22])); % 5 motor_UPDRS/output, 7-22, input  

%{
for t=1:17
    figure(t)
    plot(data(:,t))
end
%}
j=0;
k=1;

x=data(:,2:end)';
y=data(:,1)';

num=length(sub); % total number of data points
for i=1:num
    if i<=num-1
        if sub(i)==sub(i+1)
     j=j+1;
  else
      task_idx(k)=j+1;
      k=k+1;
      j=0;
  end
    else
      task_idx(k)=j+1; % single index
    end
end
T_max=size(task_idx,2); % number of tasks
%}
elseif dataset==3
    %% PD2
A=importdata('park.txt');
%te=570; % 570 1180 1970
sub=(A(1:end,1)); % 1 index
data=(A(1:end,[6,7:22])); % 6 Parkinson-Total/output, 7-22, input  

%{
for t=1:17
    figure(t)
    plot(data(:,t))
end
%}
j=0;
k=1;

x=data(:,2:end)';
y=data(:,1)';

num=length(sub); % total number of data points
for i=1:num
    if i<=num-1
        if sub(i)==sub(i+1)
     j=j+1;
  else
      task_idx(k)=j+1;
      k=k+1;
      j=0;
  end
    else
      task_idx(k)=j+1; % single index
    end
end
T_max=size(task_idx,2); % number of tasks
elseif dataset==4
    %% AD1
load ADAS_data
T_max=size(X,1);
for t=1:T_max
  task_idx(t)=size(X{t},1);  
end
x=cell2mat(X)';
y=cell2mat(Y)';
%}
elseif dataset==5
    %% AD2
load MMSE_data
T_max=size(X,1);
for t=1:T_max
  task_idx(t)=size(X{t},1);  
end
x=cell2mat(X)';
y=cell2mat(Y)';    
elseif dataset==6
    %% river_flow
N=100;
data=load('river_flow.txt');
data=data(1:N,:);
T_max=8;  
idx=[0,8,16,24,32,40,48,56];
for t=1:T_max
    X{t}=data(:,idx+t)';
    Y{t}=data(:,64+t)';
    task_idx(t)=size(X{t},2);

for i=1:size(X{t},1)
[X{t}(i,:),~]=mapminmax(X{t}(i,:),0,1); % x:d*N y:1*N
end
[Y{t},PS]=mapminmax(Y{t},0,1);
    
    %figure(t)
    %plot(X{t}(2,:))
end
x=cell2mat(X)';
y=cell2mat(Y); 
else 
%% sarcos
load sarcos
x1=cell2mat(X');
T_max=7;
task_idx=2000*ones(1,T_max);
for i=1:size(x1,2)
[x(i,:),~]=mapminmax(x1(:,i)',0,1); % x:d*N y:1*N
end
for t=1:T_max
    [Y1{t},PS]=mapminmax(Y{t}',0,1);
end

y=cell2mat(Y1); 
x=x';
end
%% normalization

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
%load AD2_TaskNumCompare;
%load AD1_Compare;
%load PD2_Compare;
%RMSE_te1=[];RMSE_te2=[];RMSE_te4=[];RMSE_te5=[];RMSE_te7=[];RMSE_te8=[];RMSE_te9=[];RMSE_te10=[];

 for iter=1:10
% training /testing
training_percent=0.5;
task_num{iter} = randperm(K);  
%task_num{iter}=[1:T_max];
for i=1:K    
    % rand task sequence 
    xc{i}=xc1{task_num{iter}(i)};
    yc{i}=yc1{task_num{iter}(i)};
    
    % randn spilt training/testing 
    %{
    [ndata, D] = size(yc{i}');        %ndata????D??
    R = randperm(ndata); 
    x_tr{i}=xc{i}(R(1:round( size(xc{i},1)*training_percent )),:);
    y_tr{i}=yc{i}(R(1:round( size(xc{i},1)*training_percent )))';
    R( (1:round( size(xc{i},1)*training_percent )) ) = [];
    x_te{i}=xc{i}(R,:);
    y_te{i}=yc{i}(R)';
    %}
    
    x_tr{i}=xc{i}(1:round( size(xc{i},1)*training_percent ),:);
    y_tr{i}=yc{i}(1:round( size(xc{i},1)*training_percent ))';
    x_te{i}=xc{i}(round( size(xc{i},1)*training_percent )+1:end,:);
    y_te{i}=yc{i}(round( size(xc{i},1)*training_percent )+1:end)';
    %}
end

T_max=length(task_idx);
%% cross validation
% parameter range
% pd: k 1-6/ rho 1-5/ rho2 1-5/
% ad1: k 1-6/ rho 4-8/ rho2 4-8/
% ad2: k 1-6/ rho 4-8/ rho2 4-8/
%{
fold=5;
for i=1:K
    indices=crossvalind('Kfold',size(x_tr{i},1),fold);
    for j=1:fold
        test=(indices==j);
        train=~test;
        x_croesstr{i,j}=x_tr{i}(train,:);
        x_croesste{i,j}=x_tr{i}(test,:);
        y_croesstr{i,j}=y_tr{i}(train,:);
        y_croesste{i,j}=y_tr{i}(test,:);
    end
end

lat_var=2; 
cout=0;
for iter1=1:8 
k=iter1;
for iter2=1:8
rho=10^(-iter2); 
for iter3=1:8
rho2=10^(-iter3); 
cout=cout+1
for i=1:1
   xvtr=x_croesstr(:,i)'; 
   yvtr=y_croesstr(:,i)';
   xvte=x_croesste(:,i)'; 
   yvte=y_croesste(:,i)';
   %[yp_tr,yp_te,ytr,yte,yp,ype] = LS_ELLA(xvtr,yvtr,xvte,yvte,k, rho,rho2,1);
  %[yp_tr,yp_te,ytr,yte,yp,ype] = LS_ELLA(x_tr,y_tr,x_te,y_te,k, rho,rho2,1);
   [yp_tr,yp_te,ytr,yte,yp,ype] = CoupleDic_ELLA_WeakRe(x_tr,y_tr,x_te,y_te,k, rho,rho2, 0.1,1);
%[yp_tr,yp_te,ytr,yte,yp,ype] = CoupleDic_ELLA_WeakRe_LS(x_tr,y_tr,x_te,y_te,k, rho,rho2, 10^(-3),1);
%[yp_tr,yp_te,ytr,yte,yp,ype] = CoupleDic_ELLA(x_tr,y_tr,x_te,y_te,k, rho,rho2, 2,1);
%[yp_tr,yp_te,ytr,yte,yp,ype] = PLS_ELLA(x_tr,y_tr,x_te,y_te,k, rho, rho2,2,1);
   rmse(i) = sqrt(mean((yte - ype).^2));
end
rmse2(iter1,iter2,iter3)=mean(rmse);
end
end
end
%[a1,a2,a3]=find(rmse2==min(min(rmse2)));

for i=1:size(rmse2,1)
for j=1:size(rmse2,2)
for k=1:size(rmse2,3)
if rmse2(i,j,k)==min(min(min(rmse2)))
best_para_idx=[i j k] 
end
end
end
end
figure(1)
%}
%% parameter
k=1; % SCHOOL 1/PD1 1                   
rho=10^-5;  % fixed           
rho2=10^-5; %    
T_pool=round(T_max/2);
taskSelec=1; % 0: randon/ 1: diversity / 2: diversity++
STL=0;

%a=[3,2,1,10^-1,10^-2]; % school
%a=[3,2,1,10^-1,10^-2,10^-3]; % PD
%a=[5,4,3,2,1,10^-1]; % PD
%% proposed

zero_shot=1; % 0: both / 1: unsuper_fea / 2: model
rou=5; % shcool:0.1 \pd: 0.01 /ad:5
%rou=a(iter);

[yte{iter},ype{iter},yp_te,ytr{iter},ypr{iter},yp_tr,RMSE_tr1(iter),RMSE_te1(iter,:),task_id,ACTpT(iter)] = UnsuperFea_ELLA(x_tr,y_tr,x_te,y_te,k,rho,rho2,rou,T_pool,0,0);
%[yte{iter},ype{iter},yp_te,ytr{iter},ypr{iter},yp_tr,RMSE_tr2(iter),RMSE_te(iter,:),task_id,ACTpT(iter)] = UnsuperFea_ELLA(x_tr,y_tr,x_te,y_te,k,rho,rho2,rou,T_pool,0,1);
%[yte,ype,yp_te,ytr,ypr,yp_tr,RMSE_tr3,RMSE_te3,task_id,ACTpT] = UnsuperFea_ELLA(x_tr,y_tr,x_te,y_te,k,rho,rho2,rou,T_pool,0,2);
[yte{iter},ype{iter},yp_te,ytr{iter},ypr{iter},yp_tr,RMSE_tr4(iter),RMSE_te2(iter,:),task_id,ACTpT(iter)] = UnsuperFea_ELLA(x_tr,y_tr,x_te,y_te,k,rho,rho2,rou,T_pool,1,0);
%[yte{iter},ype{iter},yp_te,ytr{iter},ypr{iter},yp_tr,RMSE_tr5(iter),RMSE_te(iter,:),task_id,ACTpT(iter)] = UnsuperFea_ELLA(x_tr,y_tr,x_te,y_te,k,rho,rho2,rou,T_pool,1,1);
%[yte,ype,yp_te,ytr,ypr,yp_tr,RMSE_tr6,RMSE_te6,task_id,ACTpT] = UnsuperFea_ELLA(x_tr,y_tr,x_te,y_te,k,rho,rho2,rou,T_pool,1,2);
para_UFELLA=[k,rho,rho2,rou];
%}
iter
%% active ELLA

%rho=10^-1; % school
%rho=10^-2;rho2=10^-1; % PD
%rho2=10^-4; % AD2 
%[yte{iter},ype{iter},yp_te,ytr{iter},ypr{iter},yp_tr,RMSE_tr7(iter),RMSE_te(iter,:),ACTpT(iter)] = Activef_ELLA(x_tr,y_tr,x_te,y_te,k,rho,rho2,T_pool,0);
%[yte{iter},ype{iter},yp_te,ytr{iter},ypr{iter},yp_tr,RMSE_tr8(iter),RMSE_te(iter,:),ACTpT(iter)] = Activef_ELLA(x_tr,y_tr,x_te,y_te,k,rho,rho2,T_pool,1);
%[yte{iter},ype{iter},yp_te,ytr{iter},ypr{iter},yp_tr,RMSE_tr9(iter),RMSE_te(iter,:),ACTpT(iter)] = Activef_ELLA(x_tr,y_tr,x_te,y_te,k,rho,rho2,T_pool,2);
para_ELLA=[k,rho,rho2];
%}

%% STL
%[yp_tr1,yp_te1,ytr1,yte1,yp1,ype1,ACTpT(iter)] = STL_LS(x_tr,y_tr,x_te,y_te); STL=1;
if STL==1
for i=1:T_pool
    yp_tr2{i}=yp_te1{i};
    y_tr1{i}=y_te{i};
end
for i=1:T_max-T_pool
    yp_te2{i}=yp_te1{i+T_pool};
    y_te1{i}=y_te{i+T_pool};
end
ypr{iter} = cell2mat(yp_tr2')';
ype{iter} = cell2mat(yp_te2')';
ytr{iter} = cell2mat(y_tr1')';
yte{iter} = cell2mat(y_te1')';

RMSE_tr(iter) = sqrt(mean((ytr{iter} - ypr{iter}).^2));
RMSE_te(iter) = sqrt((mean((yte{iter} - ype{iter}).^2)));
end

%RMSE_te(iter) = sqrt((mean((yte{iter} - ype{iter}).^2)));
%R_2(iter)=1-(norm(yte{iter} - ype{iter})^2/norm(yte{iter}-mean(yte{iter}))^2)
%RMSE_te=RMSE_te7;
 end
 %plot(RMSE_te1,'b')
 %hold on
 %plot(RMSE_te2,'r')
% mean(ACTpT)
task_num{iter}(1:6)
task_id
re=mean(RMSE_te);

re(end)
hold on
plot(re);
hold on

%% result
%RMSE_tr = sqrt(mean((ytr - ypr).^2));
%RMSE_te = sqrt((mean((yte - ype).^2)));
%{
mean(RMSE_te1)
mean(RMSE_te2)
mean(RMSE_te4)
mean(RMSE_te5)
mean(RMSE_te7)
mean(RMSE_te8)
mean(RMSE_te9)
mean(RMSE_te10)
%}
%std(RMSE_te7)
%RMSE_te1(end)
%RMSE_te2(end)
%RMSE_te3(end)
%RMSE_te4(end)
%RMSE_te5(end)
%RMSE_te6(end)
%RMSE_te7(end)
%RMSE_te8(end)
%RMSE_te9(end)
%}
%{
figure(1)
plot(RMSE_te1,'b')
hold on
plot(RMSE_te2,'r')
hold on
plot(RMSE_te4,'k')
hold on
plot(RMSE_te5,'g')
hold on
%}
%{
figure(2)
plot(RMSE_te1,'b')
hold on
plot(RMSE_te4,'r')
hold on
plot(RMSE_te7)
hold on
plot(RMSE_te8)
hold on
plot(RMSE_te9)

%}


%}
%% SAVE
%save ('AD2_TaskNumCompare','RMSE_te1','RMSE_te7','task_num');
%save ('AD2_rouCompare','RMSE_te1','RMSE_te2','RMSE_te4','RMSE_te5','a');
%{
save ('AD2_Compare','RMSE_te1','RMSE_te2','RMSE_te4','RMSE_te5','RMSE_te7','RMSE_te8','RMSE_te9','RMSE_te10',...
    'RMSE_tr1','RMSE_tr2','RMSE_tr4','RMSE_tr5','RMSE_tr7','RMSE_tr8','RMSE_tr9','RMSE_tr10',...
    'para_UFELLA','para_ELLA','task_num');
%}
%save ('S_UFELLA','ytr','ypr','yte','ype','RMSE_tr1','RMSE_te','ACTpT','para_UFELLA','task_num');
%save ('S_UFELLA_TF','ytr','ypr','yte','ype','RMSE_te','ACTpT','para_UFELLA','task_num');
%save ('S_UFELLAt','ytr','ypr','yte','ype','RMSE_te','ACTpT','para_UFELLA','task_num');
%save ('S_UFELLAt_TF','ytr','ypr','yte','ype','RMSE_te','ACTpT','para_UFELLA','task_num');
%save ('PD2_ELLA','ytr','ypr','yte','ype','RMSE_te','ACTpT','para_ELLA','task_num');
%save ('PD2_ELLA2','ytr','ypr','yte','ype','RMSE_te','ACTpT','para_ELLA','task_num');
%save ('PD2_ELLA3','ytr','ypr','yte','ype','RMSE_te','ACTpT','para_ELLA','task_num');
%save ('S_STL','ytr','ypr','yte','ype','RMSE_te','ACTpT','task_num');

