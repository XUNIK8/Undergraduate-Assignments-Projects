clc;clear;close all;
%% ������ģ��
%����ģ�͵����˲���
% theta��d��a��alphasigma
L(1)=Link([0 151.9 0 pi/2 0]);%��1
L(1).offset=pi/2;%����1��ʼλ��
L(2)=Link([0 86.85 -243.65 0 0]);%��2
L(2).offset=-pi/2;%����2��ʼλ��
L(3)=Link([0 -92.85 -213 0 0]);%��3
L(4)=Link([0 83.4 0 pi/2 0]);%��4
L(4).offset=-pi/2;%��4��ʼλ��
L(5)=Link([0 83.4 0 -pi/2 0]);%��5
L(5).offset=-pi/2;%��5��ʼλ��
L(6)=Link([0 83.4 0 0 0]);%��6
UR3 = SerialLink(L, 'name' , 'UR3������ģ��'); 

UR3.base= transl(0 ,0 ,-300);%����������ϵ
UR3.display();  %��ʾ�����Ļ����˵�DH����
UR3.plotopt = {'workspace',[-800,800,-2*800,800,-800,800],'tilesize',400};  %����ģ�Ϳռ��С����ש��С
view(3)
UR3.teach; 
hold on;
%%
%����
%���Բ��λ��
x=0;y=-650;z=100;
%�뾶
r=400;
h1 = makeSphere(r,x,y,z,1000);%����
hold on;
h1.EdgeColor = [0,1,1];%�ߵ���ɫ
h1.FaceColor = [1,1,0];%�����ɫ

%% ��һ�ι켣��ɽ ���� �������£�
x0=[];
y0=[];
z0=[];
for qd=-10:1:20 %���廭���߽Ƕȷ�Χ
    cd=deg2rad(17.5);%���ҽǶ�
    qd=deg2rad(qd);%���½Ƕ�
    %�����ϵ�����λ��
    z1=z+r*sin(qd);
    y1=y+r*cos(qd)*cos(cd);
    x1=x+r*cos(qd)*sin(cd);
    %���ھ���
    x0=[x0;x1];
    y0=[y0;y1];
    z0=[z0;z1];
    %���������ͼ
    plot3(x1,y1,z1,'r.')
end
L=[x0 y0 z0];%λ��������ϳ���������
for i=1:length(L)
    T(:,:,length(L)+1-i)=transl(L(i,:))*trotx(pi/2);%��λ�������λ�˾���
end
q1=UR3.ikunc(T);%�����
UR3.plot(q1)%�����˶�ͼ

%% �ڶ��ι켣��ɽ ���� �������£�
%���ÿվ���
x0=[];
y0=[];
z0=[];
for qd=-10:1:10 %���廭���߽Ƕȷ�Χ
    cd=deg2rad(30);%���ҽǶ�
    qd=deg2rad(qd);%���½Ƕ�
    %�����ϵ�����λ��
    z1=z+r*sin(qd);
    y1=y+r*cos(qd)*cos(cd);
    x1=x+r*cos(qd)*sin(cd);
    %���ھ���
    x0=[x0;x1];
    y0=[y0;y1];
    z0=[z0;z1];
    %���������ͼ
    plot3(x1,y1,z1,'r.')
end
L=[];T=[];q1=[];
L=[x0 y0 z0];%λ��������ϳ���������
for i=length(L):-1:1
    T(:,:,length(L)+1-i)=transl(L(i,:))*trotx(pi/2);%��λ�������λ�˾���
end
q1=UR3.ikcon(T);%�����
UR3.plot(q1)%�����˶�ͼ

%% �����ι켣��ɽ �º� �������ң�
%���ÿվ���
x0=[];
y0=[];
z0=[];
for cd=30:-1:5%���廭���߽Ƕȷ�Χ
    cd=deg2rad(cd);%���ҽǶ�
    qd=deg2rad(-10);%���½Ƕ�
    %�����ϵ�����λ��
    z1=z+r*sin(qd);
    y1=y+r*cos(qd)*cos(cd);
    x1=x+r*cos(qd)*sin(cd);
    %���ھ���
    x0=[x0;x1];
    y0=[y0;y1];
    z0=[z0;z1];
    %���������ͼ
    plot3(x1,y1,z1,'r.')
end
L=[];T=[];q1=[];
L=[x0 y0 z0];%λ��������ϳ���������
for i=1:length(L)
    T(:,:,i)=transl(L(i,:))*trotx(pi/2);%��λ�������λ�˾���
end
q1=UR3.ikcon(T);%�����
UR3.plot(q1)%�����˶�ͼ
%% ���Ķι켣��ɽ ���� �������£���ע��ͬ�ϣ�
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
%% ����ι켣���� ���� �������ң���ע��ͬ�ϣ�
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
%% �����ι켣���� ��Ʋ��һ���� �������£���ע��ͬ�ϣ�
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

%% ���߶ι켣���� ��Ʋ�ڶ����� �������£���ע��ͬ�ϣ�
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
%% �ڰ˶ι켣���� ���� �������£���ע��ͬ�ϣ�
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

%% ������
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