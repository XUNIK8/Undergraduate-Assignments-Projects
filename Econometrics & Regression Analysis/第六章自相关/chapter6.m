x=[3624.1; 4517.8; 8989.1; 11954.5; 14922.3; 16918; 18598; 21662.5; 26652; 34561; 46670; 57494.9; 67560]; 
y=[206.4; 381.4; 696; 826.5; 1027.1; 1117; 1154.4; 1356.3; 1655.3; 1957; 2366.2; 2808.6; 2899];
stats=regstats(y,x,'linear',{'r'}); 
plot(stats.r,'*','markersize',20); 
hold on 
line([1:length(stats.r)]',stats.r,'color','r','linewidth',3); 
plot([0,14],[0,0],'--','linewidth',3) 
xlabel('t')
ylabel('r')
b = regress(y,[ones(13,1) lagmatrix(y,1) x lagmatrix(x,1)]) 
stats3=regstats(y-b(2)*lagmatrix(y,1),x-b(2)*lagmatrix(x,1),'linear',{'tstat','beta','r'}); stats3.tstat.se 
stats3.tstat.pval 
stats3.tstat.beta 
rr=stats3.r(2:13,:); 
dw=(norm(diff(rr)))^2/(norm(rr))^2;