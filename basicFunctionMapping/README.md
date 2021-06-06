# machineLearningDataSc

1(a): Create the Training Input Data: Create a [1000 x 3] matrix of random numbers uniformly distributed between -1 and 1.
1(b): create training output data using the function 𝑦𝑇𝑟𝑢𝑒 = −10 log(𝑥_1^2) − 15 log(𝑥_2^2) − 7.5 log(𝑥_3^2) + 2
1(c): add gaussian noise using mu = 0, std = 10 to the y_true error.
1(d): repeat 1a to 1c to create test input and test output.
1(e): create LinearRegressionParameters.m that calculates the parameters (theta) of a LR
1(f): create LinearRegressionPrediction.m that calculates the predictions of an input training matrix based on the parameters calculated in (d)
1(g): create LR_Error.m that calculates the MSE of LR prediction
1(h): calculate error on above data on the below basis functions
𝜙_𝐴(𝑥_1, 𝑥_2, 𝑥_3) = [1, 𝑥_1, 𝑥_2, 𝑥_3]^𝑇
𝜙_𝐵(𝑥_1, 𝑥_2, 𝑥_3) = [1, log(𝑥_1^2) , log(𝑥_2^2) , log(𝑥_3^3)]^𝑇
