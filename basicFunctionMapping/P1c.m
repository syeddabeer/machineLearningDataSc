a=-1;
b=1;
r=(b-a).*rand(1000,3)+a;
x1=r(:,1);
x2=r(:,2);
x3=r(:,3);
ytrue=(-10*log(x1.^2))-(15*log(x2.^2))-(7.5*log(x3.^2))+2; %.^ for element wise power
rid=fopen('outputP3a.txt','w');
fprintf(rid,'\n\nFor gaussian noise:\n\n');
fprintf(rid,'\n\nBefore adding guassian noise, the ytrue vector is\n\n');
fprintf(rid,'%8.4f\n', [ytrue(1:1000,:)]);
mean=0;
stdev=10;
noise=mean+(stdev*randn(1000,1));
ytruenoisy=ytrue+noise;
fprintf('\n\nAfter adding guassian noise, the ytruenoisy vector is\n\n');
fprintf('%9.4f\n', [ytruenoisy(1:1000,:)]);
fclose(rid);