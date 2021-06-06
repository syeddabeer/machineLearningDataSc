%%%%%%%%%%%%%%Train dataset%%%%%%%%%%%%%%
a=-1;
b=1;
r=(b-a).*rand(1000,3)+a;
x1=r(:,1);
x2=r(:,2);
x3=r(:,3);
ytrue=(-10*log(x1.^2))-(15*log(x2.^2))-(7.5*log(x3.^2))+2; %.^ for element wise power
fprintf('\n\nBefore adding guassian noise, the ytrue vector is\n\n');
fprintf('%8.4f\n', [ytrue(1:1000,:)]);
mean=0;
stdev=10;
noise=mean+(stdev*randn(1000,1));
ytruenoisy=ytrue+noise;
fprintf('\n\nAfter adding guassian noise, the ytruenoisy vector is\n\n');
fprintf('%9.4f\n', [ytruenoisy(1:1000,:)]);
Xtrain = r(:, :);
ytrain = ytruenoisy(:, 1);
m = length(ytrain);

%%%%%%%%%%%%%%%%%for phi-B
x1phib=log((r(:,1)).^2);
x2phib=log((r(:,2)).^2);
x3phib=log((r(:,3)).^2);
ytruephib=(-10*log(x1phib.^2))-(15*log(x2phib.^2))-(7.5*log(x3phib.^2))+2; %.^ for element wise power
meanphib=0;
stdevphib=10;
noisephib=meanphib+(stdevphib*randn(1000,1));
ytruenoisyphib=ytruephib+noisephib;
fprintf('\n\nAfter adding guassian noise, the ytruenoisyphib vector is\n\n');
fprintf('%9.4f\n', [ytruenoisyphib(1:1000,:)]);
Xtrainphib = log((r(:, :)).^2);
ytrainphib = ytruenoisyphib(:, 1);
XtrainWithOnesphib =  [ones(m, 1) Xtrainphib];
mytrainphib = length(ytrainphib);


%%%%%%%%%%%%%%Test dataset%%%%%%%%%%%%%%
atest=-1;
btest=1;
rtest=(b-a).*rand(1000,3)+a;
x1test=rtest(:,1);
x2test=rtest(:,2);
x3test=rtest(:,3);
ytruetest=(-10*log(x1test.^2))-(15*log(x2test.^2))-(7.5*log(x3test.^2))+2; %.^ for element wise power
meantest=0;
stdevtest=10;
noisetest=meantest+(stdevtest*randn(1000,1));
ytruenoisytest=ytruetest+noisetest;
fprintf('\n\nAfter adding guassian noise, the ytruenoisytest vector is\n\n');
fprintf('%9.4f\n', [ytruenoisytest(1:1000,:)]);
Xtest = rtest(:, :);
ytest = ytruenoisytest(:, 1);
XtestWithOnes =  [ones(m, 1) Xtest];
mytest = length(ytest);

%%%%%%%%%%%%%%%%%for phi-B
x1testphib=log((rtest(:,1)).^2);
x2testphib=log((rtest(:,2)).^2);
x3testphib=log((rtest(:,3)).^2);
ytruetestphib=(-10*log(x1testphib.^2))-(15*log(x2testphib.^2))-(7.5*log(x3testphib.^2))+2; %.^ for element wise power
meantestphib=0; 
stdevtestphib=10;
noisetestphib=meantestphib+(stdevtestphib*randn(1000,1));
ytruenoisytestphib=ytruetestphib+noisetestphib;
fprintf('\n\nAfter adding guassian noise, the ytruenoisytestphib vector is\n\n');
fprintf('%9.4f\n', [ytruenoisytestphib(1:1000,:)]);
Xtestphib = log((rtest(:, :)).^2);
ytestphib = ytruenoisytestphib(:, 1);
XtestWithOnesphib =  [ones(m, 1) Xtestphib];
mytestphib = length(ytestphib);

%%%%%%%%%%Apply Basis function expansion%%%%%%%
XtrainWithOnes =  [ones(m, 1) Xtrain];
Q1 = [ones(m, 1) Xtrain];
logsquareXtrain = log(Xtrain.^2)
Q2 = [ones(m, 1) logsquareXtrain];

%%%%%%%%%%Parameters from the normal equation%%%
theta1 = LinearRegressionParameters(Q1, ytruenoisy); 
theta2 = LinearRegressionParameters(Q2, ytruenoisyphib); 

%%%%%%%%%%Display normal equation's result
fprintf('\nPart (e)\n\nTheta computed for linear function: \n');
fprintf(' %f \n\n', theta1);

fprintf('\nPart (e)\n\nTheta computed for log square function: \n');
fprintf(' %f \n\n', theta2);

%%%%%%%%%%Predictions for Test Data
PredictionMatrix1test = LinearRegressionPrediction(XtestWithOnes, theta1); 
PredictionMatrix2test = LinearRegressionPrediction(XtestWithOnesphib, theta2); 

%%%%%%%%%%Mean Square Error between predictions of Test Data and actual ytrue for Test Data
MSE1 = LRError(ytruenoisytest,PredictionMatrix1test);
MSE2 = LRError(ytruenoisytestphib,PredictionMatrix2test);

fprintf('\nMean square Error for linear function is:\n');
fprintf('%9.4f\n', MSE1);

fprintf('\nMean square Error for log square basis function is:\n');
fprintf('%9.4f\n', MSE2);
