clear;
clc;
close all;

data=xlsread('data.xlsx');

[n,m]=size(data);
Y=data(:,end);
X=data(:,1:(end-1));
X=[ones(n,1),X];
beta=inv(X'*X)*(X'*Y);

p=zeros(m,1);
tss=(Y-mean(Y))'*(Y-mean(Y));
rss=Y'*Y-beta'*X'*Y;
R=(tss-rss)/tss;
F=((tss-rss)/(m-1))/(rss/(n-m));
p(1)=1-fcdf(F,m-1,n-m);

C=diag(inv(X'*X));
sigma=sqrt(rss/(n-m));
t=zeros(m-1,1);
for i=1:(m-1)
    t(i)=beta(i+1)/(sqrt(C(i+1))*sigma);
    p(i+1)=2*(1-tcdf(abs(t(i)),n-m));
end

X1=[10821,142;11720,144];Y1=[8975;10287];
plot3(X(:,2),X(:,3),X*beta,'b');hold on;
plot3(X1(:,1),X1(:,2),[ones(2,1),X1]*beta,'g');hold on;
plot3([X(:,2);X1(:,1)],[X(:,3);X1(:,2)],[Y;Y1],'ro');grid on
legend('训练集：2004-2016年','预测集：2017-2018年','真实值');
ylabel('消费价格指数');
xlabel('人均可支配收入');
zlabel('人均消费支出');
set(0,'defaultfigurecolor','w') 
