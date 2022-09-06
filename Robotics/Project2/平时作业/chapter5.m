clc;clear;close all;
%% 构建机器人
L1=Link([0 138 0 pi/2 0],'mod');
L(1).qlim=[-pi/2, pi/2];
L2=Link([0 0 135 0 0],'mod');L2.offset=-pi/4;
L3=Link([0 0 147 0 0],'mod');L3.offset=pi/2;
robot = SerialLink([L1,L2,L3], 'name' , '机器人');  
robot.base= transl(0 ,0 ,0);
robot.display();  
space=300;
view(3)
robot.plotopt = {'workspace',[-space,space,-2*space,space,-space,space],'tilesize',space};  
robot.teach;      
%% （1）
% 起始点位姿
T1=transl(200, 120, 40);
%终止点位姿
T2=transl(220,-150,220);

t=[0:0.1:5]';
Ts1=ctraj(T1,T2,length(t));
q_s1=robot.ikine(Ts1,'mask',[1 1 1 0 0 0]);
q = [q_s1]
x=squeeze(Ts1(1,4,:)); y=squeeze(Ts1(2,4,:)); z=squeeze(Ts1(3,4,:));
x2=[x];y2=[y];z2=[z];
plot3(x2,y2,z2);
robot.plot(q);
%% （2）
N = (0:1:100)'; 
center = [175 0 5];
radius = 50;
theta = ( N/N(end) )*2*pi;
points = (center - radius*[cos(theta) sin(theta) zeros(size(theta))])';  
plot3(points(1,:),points(2,:),points(3,:),'r');
%初始状态
robot.plot([0 0 0]);

T = transl(points');
q = robot.ikine(T,'mask',[1 1 1 0 0 0]);
hold on;
robot.plot(q,'tilesize',300)
%% （3）画了一根棒棒糖
N = (0:1:50)'; 
center = [175 0 5];
radius = 10;
theta = ( N/N(end) )*2*pi;
points = (center - radius*[cos(theta) sin(theta) zeros(size(theta))])';  
plot3(points(1,:),points(2,:),points(3,:),'g');
%初始状态
T = transl(points');
q = robot.ikine(T,'mask',[1 1 1 0 0 0]);
hold on;
robot.plot(q,'tilesize',300)

N = (0:1:75)'; 
center = [175 0 5];
radius = 30;
theta = ( N/N(end) )*2*pi;
points = (center - radius*[cos(theta) sin(theta) zeros(size(theta))])';  
plot3(points(1,:),points(2,:),points(3,:),'b');
%初始状态
T = transl(points');
q = robot.ikine(T,'mask',[1 1 1 0 0 0]);
hold on;
robot.plot(q,'tilesize',300)

% 起始点位姿
T1=transl(125, 0, 50);
%终止点位姿
T2=transl(50,0,50);

t=[0:0.2:5]';
Ts1=ctraj(T1,T2,length(t));
q_s1=robot.ikine(Ts1,'mask',[1 1 1 0 0 0]);
q = [q_s1]
x=squeeze(Ts1(1,4,:)); y=squeeze(Ts1(2,4,:)); z=squeeze(Ts1(3,4,:));
x2=[x];y2=[y];z2=[z];
plot3(x2,y2,z2);
robot.plot(q);