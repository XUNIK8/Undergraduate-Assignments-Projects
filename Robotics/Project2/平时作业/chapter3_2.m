clc;clear;close all;

% %机器人参数
DR = pi/180;
%构建机器人
L1=4;L2=3;L3=2;
alp(1)=0;a(1)=L1;d(1)=0;th(1)=90;
alp(2)=0;a(2)=L2;d(2)=0;th(2)=90;
alp(3)=0;a(3)=L3;d(3)=0;th(3)=90;
L(1)=Link([th(1),d(1),a(1),alp(1),0],'mod');
L(2)=Link([th(2),d(2),a(2),alp(2),0],'mod');
L(3)=Link([th(3),d(3),a(3),alp(3),0],'mod');
ThreeR =SerialLink(L);
ThreeR.display()
%初始角度
theta = [10 20 30]'*DR;
theta1=transpose(theta);
%初速度和力矩
v0=[0.2;-0.3;-0.2];
W0=[1 2 3]';
J1=zeros(3,3);
%计算雅可比矩阵
J1=calculate_Jac(theta);
%初始关节角速度
thetadot=inv(J1)*v0;
thetadot1=transpose(thetadot);
%初始位姿矩阵
T=ThreeR.fkine(transpose(theta));
pos=transl(T);
pos(3)=(theta(1)+theta(2)+theta(3))*180/pi;
pos1=pos;
Jdet(1)=det(J1);
%计算初始转矩
tau=transpose(J1)*W0;
taua1=transpose(tau);

for i=1:50
    %新角度
    theta=theta+thetadot*0.1;
    %新theta矩阵
    theta1=cat(1,theta1,transpose(theta));
    %新雅可比矩阵
    J=calculate_Jac(theta);
    %新pos矩阵
    T=ThreeR.fkine(transpose(theta));
    pos=transl(T);
    pos(3)=(theta(1)+theta(2)+theta(3))*180/pi;
    pos1=cat(1,pos1,pos);
    %新角速度
    thetadot=inv(J)*v0;
    thetadot1=cat(1,thetadot1,transpose(thetadot));
    %新雅可比行列式
    Jdet(i+1)=det(J);
    %新转矩
    tau=transpose(J)*W0;
    %tau矩阵
    taua1=cat(1,taua1,transpose(tau));
end

% T_last最后一步位姿矩阵
disp(T)

%% 画图
t = [0:0.1:5];
% 图1
figure('color','w');
subplot(2,2,1);
plot(t,thetadot1(:,1),'r',t,thetadot1(:,2),'b',t,thetadot1(:,3),'.');
xlabel('time(s)');
ylabel('Angular veriocity(degrees/s)');
grid on;title('关节角速度随时间变化'); legend('w1','w3','w3'); 
% 图2
subplot(2,2,2);
plot(t,theta1(:,1),'r',t,theta1(:,2),'b',t,theta1(:,3),'.');
xlabel('time(s)');
ylabel('Angle(degrees)');
grid on;title('关节转角随时间变化');legend('\theta_1','\theta_2','\theta_3'); 
% 图3
subplot(2,2,3);
plot(t,pos1(:,1),'r',t,pos1(:,2),'b',t,pos1(:,3),'.');
xlabel('time(s)');
ylabel('displacement(m)');
grid on; title('机器人末端x，y坐标和绕z轴转角随时间变化');legend('X','Y','\theta_z');

%% 计算Jacobi矩阵
function [Jac]=calculate_Jac(theta);
syms	C1 C2 C3 C4 C5 C6 real;
syms	S1 S2 S3 S4 S5 S6 real;
syms	C123 S123 C12 S12 C23 S23 real;
syms	l1 l2 l3	;
Jac = [
    C123*(S23*l1+S3*l2)-S123*(C23*l1+C3*l2+l3) C123*S3*l2-S123*(C3*l2+l3) -(S123*l3);
    S123*(S23*l1+S3*l2)+C123*(C23*l1+C3*l2+l3) S3*S123*l2+C123*(C3*l2+l3)   C123*l3;
    1 1 1];

Jac=subs(Jac,[C1;C2	;C3],[cos(theta(1));cos(theta(2));cos(theta(3));]);
Jac=subs(Jac,[S1;S2	;S3],[sin(theta(1));sin(theta(2));sin(theta(3))]);
Jac=subs(Jac,[l1;l2	;l3],[4;3;2]);
Jac=subs(Jac,[C123;S123;C12;S12;C23;S23],[cos(theta(1)+ theta(2)+ theta(3));sin(theta(1)+ theta(2)+ theta(3));cos(theta(1)+ theta(2));	sin(theta(1)+ theta(2));	cos(theta(2)+ theta(3));	sin(theta(2)+ theta(3))]);
Jac=eval(Jac);
end