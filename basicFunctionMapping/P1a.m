a=-1;
b=1;
r=(b-a).*rand(1000,3)+a;
rid=fopen('outputP1a.txt','w');
fprintf(rid,'\n\nTraining Input Data (matrix):\n\n\n');
fprintf(rid, '%7.4f %7.4f %7.4f\n', [r(1:1000,:)]);
fclose(rid);
