clc;clear;close all;
% D-H参数
theta(1) = 0; d(1) = 0; a(1) = 0; alp(1) = 0;
theta(2) = 0; d(2) = 0; a(2) = 4; alp(2) = 0;
theta(3) = 0; d(3) = 0; a(3) = 3; alp(3) = 0;
L(1) = Link([theta(1), d(1), a(1), alp(1)], 'modified');
L(2) = Link([theta(2), d(2), a(2), alp(2)], 'modified');
L(3) = Link([theta(3), d(3), a(3), alp(3)], 'modified');

L(1).m = 20; 
L(2).m = 15; 
L(3).m = 10;

L(1).r = [2 0 0];
L(2).r = [1.5 0 0];
L(3).r = [1 0 0];

L(1).I = [0 0 0; 0 0 0; 0 0 0.5];
L(2).I = [0 0 0; 0 0 0; 0 0 0.2];
L(3).I = [0 0 0; 0 0 0; 0 0 0.1];

L(1).Jm = 0;
L(2).Jm = 0;
L(3).Jm = 0;
% 构建机器人
robot = SerialLink(L); 
robot.name = 'robot';
q0 = [-60,90,30]*pi/180;
%显示模型
view(3)
robot.plot(q0)
qd0 = [0,0,0];
tor = [20,5,1]';

q1 = [q0(1)];
q2 = [q0(2)];
q3 = [q0(3)];
qd1 = [qd0(1)];
qd2 = [qd0(2)];
qd3 = [qd0(3)];
G0 = robot.gravload([0,0,0]);
disp(robot.inertia([0,0,0]));
disp(robot.gravload([0,0,0]));
C = robot.coriolis(q0,qd0);

for i=0:0.01:4
    M_inv = inv(robot.inertia(q0));
    C = robot.coriolis(q0,qd0);
    G = robot.gravload(q0);
    qdd = M_inv*(tor-C*qd0'-G');
    qd = qd0 + qdd'*0.01;
    q = q0 + qd0*0.01+0.5*qdd'*0.01*0.01;
    qd0 = qd;
    q0 = q;
    q1(end+1) = q0(1);
    q2(end+1) = q0(2);
    q3(end+1) = q0(3);
    qd1(end+1) = qd0(1);
    qd2(end+1) = qd0(2);
    qd3(end+1) = qd0(3);
end
%% 画图
t = 0:0.01:4;
%图1
figure('color','w');
subplot(1,2,1);
plot(t,q1(1:401),'r',t,q2(1:401),'b',t,q3(1:401),'g');
xlabel('Time(s)');
ylabel('Angle (rad)');
grid on;title('角度曲线');legend('theta1','theta2','theta3');
%图2
subplot(1,2,2);
plot(t,qd1(1:401),'r',t,qd2(1:401),'b',t,qd3(1:401),'g');
xlabel('Time(s)');
ylabel('Angular acceleration (rad/s^2)');
grid on;title('角速度曲线');legend('w1','w2','w3');
