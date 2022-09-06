clc;clear;close all;
%% 机器人模型
%建立模型得连杆参数
% theta；d；a；alphasigma
L(1)=Link([0 151.9 0 pi/2 0]);%杆1
L(1).offset=pi/2;%连杆1初始位置
L(2)=Link([0 86.85 -243.65 0 0]);%杆2
L(2).offset=-pi/2;%连杆2初始位置
L(3)=Link([0 -92.85 -213 0 0]);%杆3
L(4)=Link([0 83.4 0 pi/2 0]);%杆4
L(4).offset=-pi/2;%杆4初始位置
L(5)=Link([0 83.4 0 -pi/2 0]);%杆5
L(5).offset=-pi/2;%杆5初始位置
L(6)=Link([0 83.4 0 0 0]);%杆6
UR3 = SerialLink(L, 'name' , 'UR3机器人模型'); 

UR3.base= transl(0 ,0 ,-300);%调整基坐标系
UR3.display();  %显示建立的机器人的DH参数
UR3.plotopt = {'workspace',[-800,800,-2*800,800,-800,800],'tilesize',400};  %设置模型空间大小、地砖大小
view(3)
UR3.teach; 
hold on;
%%
%画球
%求得圆心位置
x=0;y=-650;z=100;
%半径
r=400;
h1 = makeSphere(r,x,y,z,1000);%画球
hold on;
h1.EdgeColor = [0,1,1];%边的颜色
h1.FaceColor = [1,1,0];%面的颜色

%% 第一段轨迹（山 中竖 由上往下）
x0=[];
y0=[];
z0=[];
for qd=-10:1:20 %定义画的线角度范围
    cd=deg2rad(17.5);%左右角度
    qd=deg2rad(qd);%上下角度
    %求球上的坐标位置
    z1=z+r*sin(qd);
    y1=y+r*cos(qd)*cos(cd);
    x1=x+r*cos(qd)*sin(cd);
    %属于矩阵
    x0=[x0;x1];
    y0=[y0;y1];
    z0=[z0;z1];
    %绘制坐标点图
    plot3(x1,y1,z1,'r.')
end
L=[x0 y0 z0];%位置坐标组合成整体坐标
for i=1:length(L)
    T(:,:,length(L)+1-i)=transl(L(i,:))*trotx(pi/2);%求位置坐标的位姿矩阵
end
q1=UR3.ikunc(T);%求逆解
UR3.plot(q1)%绘制运动图

%% 第二段轨迹（山 左竖 由上往下）
%设置空矩阵
x0=[];
y0=[];
z0=[];
for qd=-10:1:10 %定义画的线角度范围
    cd=deg2rad(30);%左右角度
    qd=deg2rad(qd);%上下角度
    %求球上的坐标位置
    z1=z+r*sin(qd);
    y1=y+r*cos(qd)*cos(cd);
    x1=x+r*cos(qd)*sin(cd);
    %属于矩阵
    x0=[x0;x1];
    y0=[y0;y1];
    z0=[z0;z1];
    %绘制坐标点图
    plot3(x1,y1,z1,'r.')
end
L=[];T=[];q1=[];
L=[x0 y0 z0];%位置坐标组合成整体坐标
for i=length(L):-1:1
    T(:,:,length(L)+1-i)=transl(L(i,:))*trotx(pi/2);%求位置坐标的位姿矩阵
end
q1=UR3.ikcon(T);%求逆解
UR3.plot(q1)%绘制运动图

%% 第三段轨迹（山 下横 由左往右）
%设置空矩阵
x0=[];
y0=[];
z0=[];
for cd=30:-1:5%定义画的线角度范围
    cd=deg2rad(cd);%左右角度
    qd=deg2rad(-10);%上下角度
    %求球上的坐标位置
    z1=z+r*sin(qd);
    y1=y+r*cos(qd)*cos(cd);
    x1=x+r*cos(qd)*sin(cd);
    %属于矩阵
    x0=[x0;x1];
    y0=[y0;y1];
    z0=[z0;z1];
    %绘制坐标点图
    plot3(x1,y1,z1,'r.')
end
L=[];T=[];q1=[];
L=[x0 y0 z0];%位置坐标组合成整体坐标
for i=1:length(L)
    T(:,:,i)=transl(L(i,:))*trotx(pi/2);%求位置坐标的位姿矩阵
end
q1=UR3.ikcon(T);%求逆解
UR3.plot(q1)%绘制运动图
%% 第四段轨迹（山 右竖 由上往下）（注释同上）
x0=[];
y0=[];
z0=[];
for qd=-10:1:10
    cd=deg2rad(5);
    qd=deg2rad(qd);

    z1=z+r*sin(qd);
    y1=y+r*cos(qd)*cos(cd);
    x1=x+r*cos(qd)*sin(cd);
    
    x0=[x0;x1];
    y0=[y0;y1];
    z0=[z0;z1];
    
    plot3(x1,y1,z1,'r.')
end
L=[];T=[];q1=[];
L=[x0 y0 z0];
for i=1:length(L)
    T(:,:,length(L)+1-i)=transl(L(i,:))*trotx(pi/2);
end
q1=UR3.ikcon(T);
UR3.plot(q1)
%% 第五段轨迹（大 横线 由左往右）（注释同上）
x0=[];
y0=[];
z0=[];
for cd=-5:-1:-30
    cd=deg2rad(cd);
    qd=deg2rad(6);

    z1=z+r*sin(qd);
    y1=y+r*cos(qd)*cos(cd);
    x1=x+r*cos(qd)*sin(cd);
    
    x0=[x0;x1];
    y0=[y0;y1];
    z0=[z0;z1];
    
    plot3(x1,y1,z1,'r.')
    end
    L=[];T=[];q1=[];
    L=[x0 y0 z0];
for i=1:length(L)
    T(:,:,i)=transl(L(i,:))*trotx(pi/2);
end
q1=UR3.ikunc(T);
UR3.plot(q1)
%% 第六段轨迹（大 左撇第一部分 由上往下）（注释同上）
x0=[];
y0=[];
z0=[];
for qd=6:1:18
    cd=deg2rad(-17.5);
    qd=deg2rad(qd);

    z1=z+r*sin(qd);
    y1=y+r*cos(qd)*cos(cd);
    x1=x+r*cos(qd)*sin(cd);
    
    x0=[x0;x1];
    y0=[y0;y1];
    z0=[z0;z1];
    plot3(x1,y1,z1,'r.')
end
L=[];T=[];q1=[];
L=[x0 y0 z0];
for i=1:length(L)
    T(:,:,length(L)+1-i)=transl(L(i,:))*trotx(pi/2);
end
q1=UR3.ikunc(T);
UR3.plot(q1)

%% 第七段轨迹（大 左撇第二部分 由上往下）（注释同上）
x0=[];
y0=[];
z0=[];
cq1=linspace(deg2rad(-5),deg2rad(-17.5),17);
for qd=-10:1:6
    cd=cq1(qd+11);
    qd=deg2rad(qd);
    
    z1=z+r*sin(qd);
    y1=y+r*cos(qd)*cos(cd);
    x1=x+r*cos(qd)*sin(cd);
    
    x0=[x0;x1];
    y0=[y0;y1];
    z0=[z0;z1];
    
    plot3(x1,y1,z1,'r.');
end
L=[];T=[];q1=[];
L=[x0 y0 z0];
for i=1:length(L)
    T(:,:,length(L)+1-i)=transl(L(i,:))*trotx(pi/2);
end
q1=UR3.ikunc(T);
UR3.plot(q1)
%% 第八段轨迹（大 右捺 由上往下）（注释同上）
x0=[];
y0=[];
z0=[];
cq1=linspace(deg2rad(-30),deg2rad(-17.5),17);
for qd=-10:1:6

    cd=cq1(qd+11);
    qd=deg2rad(qd);

    z1=z+r*sin(qd);
    y1=y+r*cos(qd)*cos(cd);
    x1=x+r*cos(qd)*sin(cd);
    
    x0=[x0;x1];
    y0=[y0;y1];
    z0=[z0;z1];
    
    plot3(x1,y1,z1,'r.');
end
L=[];T=[];q1=[];
L=[x0 y0 z0];
for i=1:length(L)
    T(:,:,length(L)+1-i)=transl(L(i,:))*trotx(pi/2);
end
q1=UR3.ikunc(T);
UR3.plot(q1)

%% 画球函数
function h = makeSphere(r, centerx, centery, centerz, N)

if nargin == 5
    [x,y,z] = sphere(N);
else
    [x,y,z] = sphere(50);
end
h = surf(r*x+centerx, r*y+centery, r*z+centerz);
h.EdgeColor = rand(1,3);
h.FaceColor = h.EdgeColor;
end