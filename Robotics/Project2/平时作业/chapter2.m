clc;clear;close all;
%% 2.1
clc;clear;close all;
disp('Question2.1:')
P = [1;2;3]';
R1 = rotx(30,'deg');
R2 = roty(60,'deg');
P1= (R2*R1*P')'
%% 2.2
clc;clear;close all;
disp('Question2.2:')
% 原顺序
T1=trotz(90,'deg')*transl(5,3,6)*trotx(90,'deg');
p=(T1*[5;3;4;1])'
% （2）（3）交换顺序
T2=transl(5,3,6)*trotz(90,'deg')*trotx(90,'deg');
p2=(T2*[5;3;4;1])'

%% 2.3
clc;clear;close all;
disp('Question2.3:')
%(1)
T1=[0.527 -0.574 0.628 2
    0.369 0.819 0.439 5
    -0.766 0 0.643 3
    0 0 0 1];
T1_inv=inv(T1)
%(2)
T2=[0.92 0 0.39 5
    0 1 0 6
    -0.39 0 0.92 2
    0 0 0 1];
T2_inv=inv(T2)
%% 2.4
% 答：绕o轴转-β角，绕a轴转-α角
disp('Question2.1')
disp('绕o轴转-β角，绕a轴转-α角')

%% 2.5
clc;clear;close all;
disp('Question2.5:')
%(1)
syms r b g;
eq1=r*sind(b)*cosd(g)-3.1375;
eq2=r*sind(b)*sind(g)-2.195;
eq3=r*cosd(b)-3.214;
s=vpasolve(eq1,eq2,eq3,r,b,g);
% 得到解
gamma = eval(s.g)
beta = eval(s.b)
r = eval(s.r)

%(2)
% 代入计算
T1 = [
    cosd(beta)*cosd(gamma) -sind(gamma) sind(beta)*cosd(gamma) r*sind(beta)*cosd(gamma)
    cosd(beta)*cosd(gamma) cosd(gamma) sind(beta)*sind(gamma) r*sind(beta)*sind(gamma)
    -sind(beta) 0 cosd(beta) r*cosd(beta)
    0 0 0 1
    ]
%% 2.6
clc;clear;close all;
disp('Question2.6:')
format long g
T = [
    0.527 -0.574 0.628 4;
    0.369 0.819 0.439 6;
    -0.766 0 0.643 9;
    0 0 0 1
    ];
% 第一组解
phia1 = atan2d(T(2,1),T(1,1));
phio1 = atan2d((-T(3,1)),(T(1,1)*cosd(phia1)+T(2,1)*sind(phia1)));
phin1 = atan2d((-T(2,3)*cosd(phia1)+T(1,3)*sind(phia1)),(T(2,2)*cosd(phia1)-T(1,2)*sind(phia1)));
% 第二组解
phia2 = atan2d(-T(2,1),-T(1,1));
phio2 = atan2d((-T(3,1)),(T(1,1)*cosd(phia2)+T(2,1)*sind(phia2)));
phin2 = atan2d((-T(2,3)*cosd(phia2)+T(1,3)*sind(phia2)),(T(2,2)*cosd(phia2)-T(1,2)*sind(phia2)));
% 两组解
Answer1 = [phia1,phio1,phin1]
Answer2 = [phia2,phio2,phin2]

%% 2.7
clc;clear;close all;
disp('Question2.7:')
T = [
    0.527 -0.574 0.628 4;
    0.369 0.819 0.439 6;
    -0.766 0 0.643 9;
    0 0 0 1
    ];
% 第一组解
phi1 = atan2d(T(2,3),T(1,3));
Omiga1 = atan2d(-T(1,1)*sind(phi1)+T(2,1)*cosd(phi1),-T(1,2)*sind(phi1)+T(2,2)*cosd(phi1));
Theta1 = atan2d(T(1,3)*cosd(phi1)+T(2,3)*sind(phi1),T(3,3));
% 第二组解
phi2 = atan2d(-T(2,3),-T(1,3));
Omiga2 = atan2d(-T(1,1)*sind(phi2)+T(2,1)*cosd(phi2),-T(1,2)*sind(phi2)+T(2,2)*cosd(phi2));
Theta2 = atan2d(T(1,3)*cosd(phi2)+T(2,3)*sind(phi2),T(3,3));
% 两组解
Answer1 = [phi1,Theta1,Omiga1]
Answer2 = [phi2,Theta2,Omiga2]
%% 2.8
clc;clear;close all;
disp('Question2.8:')
%(1)
disp('(1)')
T=transl(0,6,0); 
R=trotx(pi/3);
T1=transl(0,0,3); 
R1=trotz(pi/3);
R2=trotx(pi/4);
A=R2*R1*T1*T*R

%(2)
disp('(2)')
% 第一组解
phi1 = atan2d(A(2,3),A(1,3));
Omiga1 = atan2d(-A(1,1)*sind(phi1)+A(2,1)*cosd(phi1),-A(1,2)*sind(phi1)+A(2,2)*cosd(phi1));
Theta1 = atan2d(A(1,3)*cosd(phi1)+A(2,3)*sind(phi1),A(3,3));
% 第二组解
phi2 = atan2d(-A(2,3),-A(1,3));
Omiga2 = atan2d(-A(1,1)*sind(phi2)+A(2,1)*cosd(phi2),-A(1,2)*sind(phi2)+A(2,2)*cosd(phi2));
Theta2 = atan2d(A(1,3)*cosd(phi2)+A(2,3)*sind(phi2),A(3,3));
% 两组解
Answer1 = [phi1,Theta1,Omiga1]
Answer2 = [phi2,Theta2,Omiga2]
%% 2.9
clc;clear;close all;
disp('Question2.9:')
%(1)
disp('(1)')
T=transl(5,0,0); 
R=troty(pi/3);
R1=trotz(pi/3);
T1=transl(0,0,3); 
R2=trotx(pi/4);
T=R2*R1*T1*T*R

%(2)
disp('(2)')
% 第一组解
phia1 = atan2d(T(2,1),T(1,1));
phio1 = atan2d((-T(3,1)),(T(1,1)*cosd(phia1)+T(2,1)*sind(phia1)));
phin1 = atan2d((-T(2,3)*cosd(phia1)+T(1,3)*sind(phia1)),(T(2,2)*cosd(phia1)-T(1,2)*sind(phia1)));
% 第二组解
phia2 = atan2d(-T(2,1),-T(1,1));
phio2 = atan2d((-T(3,1)),(T(1,1)*cosd(phia2)+T(2,1)*sind(phia2)));
phin2 = atan2d((-T(2,3)*cosd(phia2)+T(1,3)*sind(phia2)),(T(2,2)*cosd(phia2)-T(1,2)*sind(phia2)));
% 两组解
Answer1 = [phia1,phio1,phin1]
Answer2 = [phia2,phio2,phin2]
%% 2.10
clc;clear;close all;
disp('Question2.10:')
%(1)
disp('(1)')
T_obj=[1 0 0 1;0 0 -1 4;0 1 0 0;0 0 0 1];
T_R=[0 -1 0 2;1 0 0 -1;0 0 1 0;0 0 0 1];
T_H=inv(T_R)*T_obj
%(2)不可以，因为球坐标变换矩阵的第3行第2列元素为0；
disp('(2)')
disp('不可以，因为球坐标变换矩阵的第3行第2列元素为0')
%(3)
px = T_H(1,4)
py = T_H(2,4)
pz = T_H(3,4)

% 第一组解
phi1 = atan2d(T_H(2,3),T_H(1,3));
Omiga1 = atan2d(-T_H(1,1)*sind(phi1)+T_H(2,1)*cosd(phi1),-T_H(1,2)*sind(phi1)+T_H(2,2)*cosd(phi1));
Theta1 = atan2d(T_H(1,3)*cosd(phi1)+T_H(2,3)*sind(phi1),T_H(3,3));
% 第二组解
phi2 = atan2d(-T_H(2,3),-T_H(1,3));
Omiga2 = atan2d(-T_H(1,1)*sind(phi2)+T_H(2,1)*cosd(phi2),-T_H(1,2)*sind(phi2)+T_H(2,2)*cosd(phi2));
Theta2 = atan2d(T_H(1,3)*cosd(phi2)+T_H(2,3)*sind(phi2),T_H(3,3));
% 两组解
Answer1 = [phi1,Theta1,Omiga1]
Answer2 = [phi2,Theta2,Omiga2]
%% 2.29
clc;clear;close all;
disp('Question2.29:')
syms theta1 l1 l2 l3 l4 l5 l6
%建立坐标系
%DH参数表
%    theta    d    a    alpha
%1   theta1   l4   0      0
%2     0      0    l5     -90
%3     90     l6   0      0

%填写代入数据即可，下为通式
T1= [1 0 0 l1
    0 1 0 l2
    0 0 1 l3
    0 0 0 1];
L1=Link([0,l4,0,0,0]);
A1=[L1.A(theta1).n,L1.A(theta1).o,L1.A(theta1).a,L1.A(theta1).t;0 0 0 1];
L2=Link([0,0,l5,-pi/2,1]);
A2=[L2.A(0).n,L2.A(0).o,L2.A(0).a,L2.A(0).t;0 0 0 1];
L3=Link([0,l6,0,0,0]);
A3=[floor(L3.A(pi/2).n),floor(L3.A(pi/2).o),floor(L3.A(pi/2).a),L3.A(pi/2).t;0 0 0 1];            

T = T1*A1*A2*A3
%% 2.30 
clc;clear;close all;
disp('Question2.30:')
%对应元素相等解方程
theta = atan2(0.9563,-0.2924)*180/pi
theta1 = atan2((0.8172-0.9563),(0.6978+0.2924))*180/pi
theta2 = theta - theta1
%% 2.31
clc;clear;close all;
disp('Question2.31:')
syms theta1 theta2 theta3 d1 d2 d3 d4 d5
%建立坐标系
%DH参数表:
%    theta    d    a    alpha
%1   theta1   0    d3      0
%2   theta2   0    d4     180
%3   theta3   d5   0       0

%填写代入数据即可，下为通式
T1= [1 0 0 0
    0 1 0 0
    0 0 1 d1+d2
    0 0 0 1];
L1=Link([0,0,d3,0,0]);
A1=[L1.A(theta1).n,L1.A(theta1).o,L1.A(theta1).a,L1.A(theta1).t;0 0 0 1];
L2=Link([0,0,d4,pi,0]);
A2=[L2.A(theta2).n,L2.A(theta2).o,L2.A(theta2).a,L2.A(theta2).t;0 0 0 1];
L3=Link([0,d5,0,0,0]);
A3=[L3.A(theta3).n,L3.A(theta3).o,L3.A(theta3).a,L3.A(theta3).t;0 0 0 1];

T = T1*A1*A2*A3

%% 2.32（法1：存在由于pi/2引起的精度显示问题，看似过长的计算结果实则为0，结果也正确；为看起来舒适，引入法2）
clc;clear;close all;
disp('Question2.32:')
syms theta1 theta3 l1 l2 l3;
%建立坐标系
%DH参数表:
%    theta       d      a      alpha
%1   90+theta1   0      0        0
%2     0         l2     0       -90
%3   theta3      0      0        90
%4     0         l3     0        0

%填写代入数据即可，下为通式
T1= [1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1];
L1=Link([0,0,0,0,0]);
A1=[L1.A(theta1+pi/2).n,L1.A(theta1+pi/2).o,L1.A(theta1+pi/2).a,L1.A(theta1+pi/2).t;0 0 0 1];
L2=Link([0,l2,0,-pi/2,0]);
A2=[L2.A(0).n,L2.A(0).o,L2.A(0).a,L2.A(0).t;0 0 0 1];
L3=Link([0,0,0,pi/2,0]);
A3=[L3.A(theta3).n,L3.A(theta3).o,L3.A(theta3).a,L3.A(theta3).t;0 0 0 1];
L4=Link([0,l3,0,0,0]);
A4=[L4.A(0).n,L4.A(0).o,L4.A(0).a,L4.A(0).t;0 0 0 1];

T = T1*A1*A2*A3
%% 2.32（法2）
clc;clear;close all;
disp('Question2.32:')
syms theta1 theta3 l1 l2;
T1=transl(l1,0,0);
A1=trotz(90+theta1)*transl(0,0,0)*transl(0,0,0)*trotx(90,'deg');
A2=trotz(0)*transl(0,0,l2)*transl(0,0,0)*trotx(-90,'deg');
A3=trotz(theta3)*transl(0,0,0)*transl(0,0,0)*trotx(90,'deg');
T=T1*A1*A2*A3

%% 2.33（法1：存在由于pi/2引起的精度显示问题，看似过长的计算结果实则为0，结果也正确；为看起来舒适，引入法2）
clc;clear;close all;
disp('Question2.33:')
%建立坐标系
%DH参数表:
%    theta        d      a      alpha
%1     90         0      0      -90
%2     45         6      15       0
%3     0          0      1       -90
%4     0          18     0        90
%5    -45         0      0       -90
%6     0          0      0        0

T1= [1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1];
L1=Link([0,0,0,-pi/2,0]);
A1=[L1.A(pi/2).n,L1.A(pi/2).o,L1.A(pi/2).a,L1.A(pi/2).t;0 0 0 1];
L2=Link([0,6,15,0,0]);
A2=[L2.A(pi/4).n,L2.A(pi/4).o,L2.A(pi/4).a,L2.A(pi/4).t;0 0 0 1];
L3=Link([0,0,1,-pi/2,0]);
A3=[L3.A(0).n,L3.A(0).o,L3.A(0).a,L3.A(0).t;0 0 0 1];
L4=Link([0,18,0,pi/2,0]);
A4=[L4.A(0).n,L4.A(0).o,L4.A(0).a,L4.A(0).t;0 0 0 1];
L5=Link([0,0,0,-pi/2,0]);
A5=[L5.A(-pi/4).n,L5.A(-pi/4).o,L5.A(-pi/4).a,L5.A(-pi/4).t;0 0 0 1];
L6=Link([0,0,0,0,0]);
A6=[L6.A(0).n,L6.A(0).o,L6.A(0).a,L6.A(0).t;0 0 0 1];

T = T1*A1*A2*A3*A4*A5*A6
%% 2.33（法2）
clc;clear;close all;
disp('Question2.33:')
th1=0; th2=45; th3=0; th4=0; th5=-45; th6=0;
d2=6; a2=15; a3=1; d4=18;
T1=transl(0,0,0);
A1=trotz(90+th1,'deg')*transl(0,0,0)*transl(0,0,0)*trotx(-90,'deg');
A2=trotz(th2,'deg')*transl(0,0,d2)*transl(a2,0,0)*trotx(0,'deg');
A3=trotz(th3,'deg')*transl(0,0,0)*transl(a3,0,0)*trotx(-90,'deg');
A4=trotz(th4,'deg')*transl(0,0,d4)*transl(0,0,0)*trotx(90,'deg');
A5=trotz(th5,'deg')*transl(0,0,0)*transl(0,0,0)*trotx(-90,'deg');
A6=trotz(th6,'deg')*transl(0,0,0)*transl(0,0,0)*trotx(0,'deg');
T=T1*A1*A2*A3*A4*A5*A6

%% 2.34（法1：存在由于pi/2引起的精度显示问题，看似过长的计算结果实则为0，结果也正确；为看起来舒适，引入法2）
clc;clear;close all;
disp('Question2.33:')
%建立坐标系
%DH参数表:
%    theta       d      a      alpha
%1   theta1      l4     0       -90
%2    -90        l5     l6       90
%3     90        l7     0        90
%4   theta4      l8     0        0
syms theta1 theta4 l1 l2 l3 l4 l5 l6 l7 l8;
T1= [1 0 0 l1
    0 1 0 l2
    0 0 1 l3
    0 0 0 1];
L1=Link([0,l4,0,-pi/2,0]);
A1=[L1.A(theta1).n,L1.A(theta1).o,L1.A(theta1).a,L1.A(theta1).t;0 0 0 1];
L2=Link([0,l5,l6,pi/2,0]);
A2=[L2.A(-pi/2).n,L2.A(-pi/2).o,L2.A(-pi/2).a,L2.A(-pi/2).t;0 0 0 1];
L3=Link([0,l7,0,pi/2,0]);
A3=[L3.A(pi/2).n,L3.A(pi/2).o,L3.A(pi/2).a,L3.A(pi/2).t;0 0 0 1];
L4=Link([0,l8,0,0,0]);
A4=[L4.A(theta4).n,L4.A(theta4).o,L4.A(theta4).a,L4.A(theta4).t;0 0 0 1];

T = T1*A1*A2*A3*A4

%% 2.34（法2）
clc;clear;close all;
disp('Question2.34:')
syms th1 th4 L1 L2 L3 L4 L5 L6 L7 L8;
T1=transl(L1,L2,L3);
A1=trotz(th1,'deg')*transl(0,0,L4)*transl(0,0,0)*trotx(-90,'deg');
A2=trotz(-90,'deg')*transl(0,0,L5)*transl(L6,0,0)*trotx(90,'deg');
A3=trotz(90,'deg')*transl(0,0,L7)*transl(0,0,0)*trotx(90,'deg');
A4=trotz(th4,'deg')*transl(0,0,L8)*transl(0,0,0)*trotx(0,'deg');
T=T1*A1*A2*A3*A4
