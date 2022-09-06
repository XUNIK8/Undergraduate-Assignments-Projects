clc;clear;close all;

% 构建机器人
L(1)=Link('d',0,'a',1,'alpha',0, 'm',19.515, 'r',[-0.5,0,0], 'I', [0.0813,1.6303,1.6303]);
L(2)=Link('d',0,'a',0.5,'alpha',0, 'm',9.7575, 'r',[-0.25,0,0], 'I', [0.0406,0.8152,0.8152]);
robot = SerialLink(L);
robot.name = '机器人模型';
% 条件
ankle = [pi/18 pi/2];
q = [];
track = [];
t = [0:0.01:1]';

for i=1:102
    T0 = robot.fkine(ankle).t;
    track = [track; T0(1), T0(2)+0.005];
    T1 = transl([T0(1), T0(2)+0.005, T0(3)]);
    ankle=robot.ikine(T1,ankle, 'mask',[1, 1, 0, 0, 0, 0]);
    q = [q;ankle];
end

temp = q;
q = q(2:end,:);
% 图1
figure('color','w');
subplot(2,3,1);
plot(t,q(:,1),'r',t,q(:,2),'b');
xlabel('Time(s)')
ylabel('Angle (rad)')
grid on;title('角度');legend('theta1','theta2');
% 图2
subplot(2,3,2);
qd = diff(temp)/0.01;
plot(t,qd(:,1),'r',t,qd(:,2),'b');
xlabel('Time(s)')
ylabel('Angular verlocity (rad/s)')
grid on;title('角速度');legend('w1','w2');
% 图3
subplot(2,3,3);
temp2 = [0, 0; qd];
qdd = diff(temp2)/0.01;
plot(t,qdd(:,1),'r',t,qdd(:,2),'b');
xlabel('Time(s)')
ylabel('Angular acceleration (rad/s^2)')
grid on;title('角加速度');legend('a1','a2');
% 图4
subplot(2,3,4);
yyaxis left
plot(t,track(2:end,1),'r',t,track(2:end,2),'b-');
ylim([0.6 1.3])
xlabel('Time(s)')
ylabel('displacement(m)')
yyaxis right
plot(t,(q(:,2)+q(:,1)),'g');
ylim([1.05 1.95])
ylabel('Angle(degree)')
grid on;title('机器人末端x,y坐标和绕z轴转角随时间变化');legend('x','y','thetaZ');
% 图5
subplot(2,3,5);
tu = robot.rne(q,qd,qdd);
plot(t,tu(:,1)+180,'r');
hold on
plot(t,tu(:,2),'b');
xlabel('Time(s)')
ylabel('Torque(N/m)')
grid on;title('关节力矩');legend('T1','T2');
