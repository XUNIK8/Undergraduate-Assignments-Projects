clc;clear;close all;
%% 3.1
clc;clear;close all;
T1 = [
    0 0 1 2
    1 0 0 7
    0 1 0 5
    0 0 0 1
    ];
operator = [
    0 -0.15 0 0.1
    0.15 0 0 0.1
    0 0 0 0.3
    0 0 0 0
    ];
dT = operator*T1;
T = T1+dT
%% 3.2
clc;clear;close all;
T = [
    1 0 0 5
    0 0 1 3
    0 -1 0 8
    0 0 0 1
    ];
dT = [
    0 -0.1 -0.1 0.6
    0.1 0 0 0.5
    -0.1 0 0 -0.5
    0 0 0 0
    ];
%计算微分算子,求微分变化量
operator = dT*inv(T) 
%计算相对于T坐标系的微分算子
operatorT = inv(T)*dT 
%% 3.3
clc;clear;close all;
%(a)
% 已知微分变化量,直接写出微分算子
operator = [
    0 0 0.1 1
    0 0 0 0
    -0.1 0 0 0.5
    0 0 0 0
    ]
%(b)
% 相对于坐标系A的微分算子
A = [
    0 0 1 10
    1 0 0 5
    0 1 0 0
    0 0 0 1
    ];
operatorA = inv(A)*operator*A
%% 3.4
clc;clear;close all;
%(a)
T1 = [
    1 0 0 5
    0 0 -1 3
    0 1 0 6
    0 0 0 1
    ];
T2 = [
    1 0 0.1 4.8
    0.1 0 -1 3.5
    0 1 0 6.2
    0 0 0 1
    ];
Q =T2*inv(T1)
%(b)
% operator=Q-I
I = [
    1 0 0 0
    0 1 0 0 
    0 0 1 0
    0 0 0 1
    ];
operator = Q-I
%(c)
d=[0.1,0,0.2]
delta=[0,0,0.1]
%% 3.5
clc;clear;close all;
T = [
    0 1 0 10
    1 0 0 5
    0 0 -1 0
    0 0 0 1
    ];
J = [
    8 0 0 0 0 0
    -3 0 1 0 0 0
    0 10 0 0 0 0
    0 1 0 0 1 0
    0 0 0 1 0 0
    -1 0 0 0 0 1
    ];
D0 = [0;0.1;-0.1;0.2;0.2;0];
D = (J*D0)';
% 由微分变化量D可推得算子
operator = [
    0 0 0.2 0
    0 0 -0.3 -0.1
    -0.2 0.3 0 1
    0 0 0 0
    ];
% 坐标系变化
dT = T*operator
% 新位置
T2 = T+dT
% 新算子
operator2 = T*operator*inv(T)