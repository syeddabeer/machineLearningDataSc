a=-1;
b=1;
r=(b-a).*rand(1000,3)+a;
x1=r(:,1);
x2=r(:,2);
x3=r(:,3);
ytrue=(-10*log(x1.^2))-(15*log(x2.^2))-(7.5*log(x3.^2))+2; %.^ for element wise power
rid=fopen('outputP2a.txt','w');
fprintf(rid,'\n\nvalue of ytrue is\n');
fprintf(rid,'%8.4f\n', [ytrue(1:1000,:)]);
fclose(rid);



