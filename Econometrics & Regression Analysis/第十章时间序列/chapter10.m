clc,clear
y=[100280 110863 121717 137422 161840 187318 319438 270092 319144 348517 412119 487940 538580 592963 641280 685992 740065 820754 900309 990865 1015986];
data=y;
lenD=length(data);
a=[0.01 0.2 0.3 ]; 
lenA=length(a);
y1(1,1:lenA)=(data(1)+data(2))/2;
for i=2:lenD
y1(i,:)=a*data(i-1)+(1-a).*y1(i-1);
end
y1
 
next=a*data(lenD)+(1-a).*y1(lenD,:)


clc,clear
y=[100280 110863 121717 137422 161840 187318 319438 270092 319144 348517 412119 487940 538580 592963 641280 685992 740065 820754 900309 990865 1015986];
data=y;
lenD=length(data);
a=0.3;
st1(1)=data(1);
st2(2)=data(1);
 
for i=2:lenD
st1(i)=a*data(i)+(1-a).*st1(i-1);
st2(i)=a*st1(i)+(1-a).*st2(i-1);
end
b1=2*st1-st2
b2=a/(1-a)*(st1-st2)
y2=b1+b2


yt=[100280 110863 121717 137422 161840 187318 319438 270092 319144 348517 412119 487940 538580 592963 641280 685992 740065 820754 900309 990865 1015986];
n=length(yt); 
alpha=0.3; st1_0=mean(yt(1:3)); st2_0=st1_0;st3_0=st1_0; 
st1(1)=alpha*yt(1)+(1-alpha)*st1_0; 
st2(1)=alpha*st1(1)+(1-alpha)*st2_0; 
st3(1)=alpha*st2(1)+(1-alpha)*st3_0; 
for i=2:n 
 st1(i)=alpha*yt(i)+(1-alpha)*st1(i-1); 
 st2(i)=alpha*st1(i)+(1-alpha)*st2(i-1); 
 st3(i)=alpha*st2(i)+(1-alpha)*st3(i-1); 
end 
xlswrite('output.xls',[st1',st2',st3']) 
st1=[st1_0,st1];st2=[st2_0,st2];st3=[st3_0,st3]; 
a=3*st1-3*st2+st3; 
b=0.5*alpha/(1-alpha)^2*((6-5*alpha)*st1-2*(5-4*alpha)*st2+(4-3*alpha)*st3); 
c=0.5*alpha^2/(1-alpha)^2*(st1-2*st2+st3); 
yh=a+b+c; 
xlswrite('output.xls',yh','Sheet1','D1') 
plot(1:n,yt,'*',1:n,yh(1:n),'O') 
legend('Real','Predict') 
coe=[c(n+1),b(n+1),a(n+1)]; 
yh1=polyval(coe,2)
yh2=polyval(coe,3)
yh3=polyval(coe,4)
yh4=polyval(coe,5)
yh5=polyval(coe,6)


