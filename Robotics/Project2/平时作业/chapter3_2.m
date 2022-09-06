clc;clear;close all;

% %�����˲���
DR = pi/180;
%����������
L1=4;L2=3;L3=2;
alp(1)=0;a(1)=L1;d(1)=0;th(1)=90;
alp(2)=0;a(2)=L2;d(2)=0;th(2)=90;
alp(3)=0;a(3)=L3;d(3)=0;th(3)=90;
L(1)=Link([th(1),d(1),a(1),alp(1),0],'mod');
L(2)=Link([th(2),d(2),a(2),alp(2),0],'mod');
L(3)=Link([th(3),d(3),a(3),alp(3),0],'mod');
ThreeR =SerialLink(L);
ThreeR.display()
%��ʼ�Ƕ�
theta = [10 20 30]'*DR;
theta1=transpose(theta);
%���ٶȺ�����
v0=[0.2;-0.3;-0.2];
W0=[1 2 3]';
J1=zeros(3,3);
%�����ſɱȾ���
J1=calculate_Jac(theta);
%��ʼ�ؽڽ��ٶ�
thetadot=inv(J1)*v0;
thetadot1=transpose(thetadot);
%��ʼλ�˾���
T=ThreeR.fkine(transpose(theta));
pos=transl(T);
pos(3)=(theta(1)+theta(2)+theta(3))*180/pi;
pos1=pos;
Jdet(1)=det(J1);
%�����ʼת��
tau=transpose(J1)*W0;
taua1=transpose(tau);

for i=1:50
    %�½Ƕ�
    theta=theta+thetadot*0.1;
    %��theta����
    theta1=cat(1,theta1,transpose(theta));
    %���ſɱȾ���
    J=calculate_Jac(theta);
    %��pos����
    T=ThreeR.fkine(transpose(theta));
    pos=transl(T);
    pos(3)=(theta(1)+theta(2)+theta(3))*180/pi;
    pos1=cat(1,pos1,pos);
    %�½��ٶ�
    thetadot=inv(J)*v0;
    thetadot1=cat(1,thetadot1,transpose(thetadot));
    %���ſɱ�����ʽ
    Jdet(i+1)=det(J);
    %��ת��
    tau=transpose(J)*W0;
    %tau����
    taua1=cat(1,taua1,transpose(tau));
end

% T_last���һ��λ�˾���
disp(T)

%% ��ͼ
t = [0:0.1:5];
% ͼ1
figure('color','w');
subplot(2,2,1);
plot(t,thetadot1(:,1),'r',t,thetadot1(:,2),'b',t,thetadot1(:,3),'.');
xlabel('time(s)');
ylabel('Angular veriocity(degrees/s)');
grid on;title('�ؽڽ��ٶ���ʱ��仯'); legend('w1','w3','w3'); 
% ͼ2
subplot(2,2,2);
plot(t,theta1(:,1),'r',t,theta1(:,2),'b',t,theta1(:,3),'.');
xlabel('time(s)');
ylabel('Angle(degrees)');
grid on;title('�ؽ�ת����ʱ��仯');legend('\theta_1','\theta_2','\theta_3'); 
% ͼ3
subplot(2,2,3);
plot(t,pos1(:,1),'r',t,pos1(:,2),'b',t,pos1(:,3),'.');
xlabel('time(s)');
ylabel('displacement(m)');
grid on; title('������ĩ��x��y�������z��ת����ʱ��仯');legend('X','Y','\theta_z');

%% ����Jacobi����
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