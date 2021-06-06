##### IMPORTING PACKAGES #####
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
# pd.set_option('display.notebook_repr_html', False)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 150)
# pd.set_option('display.max_seq_items', None)
# import seaborn as sns
# sns.set_context('notebook')
# sns.set_style('white')


##### SCATTER PLOT FUNCTION DEFINITION #####
def scatterPlot(data, label_x, label_y, label_pos, label_neg, axes=None):
    # Get indexes for class 0 and class 1
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1
    # If no specific axes object has been passed, get the current axes.
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='r', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='g', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True);

##### LOADING DATA SET GIVEN IN TEXT FORMAT #####
data = np.loadtxt('LogisticRegressionData1.txt', delimiter=',')
print('Dimensions: ',data.shape)
print(data[1:6,:])

X_withoutaugment = data[:,0:2]
X = np.c_[np.ones((data.shape[0],1)), X_withoutaugment]
y = np.c_[data[:,2]]


print("\n DATA PLOT BEFORE LOGISTIC REGRESSION\n")
print("========================================\n\n")
scatterPlot(data, 'Feature1', 'Feature2', 'Positive Class', 'Negative Class')
plt.show()


##### HYPOTHESIS FUNCTION IS A SIGMOID FOR LOGISTIC REGRESSION #####
def sigmoid(wtx):
    return(1 / (1 + np.exp(-wtx)))


##### USING THE COST FUNCTION EQUATION FOR THE LOGISTIC REGRESSION
def costFunction(weight, X, y):
    m = y.size
    h = sigmoid(X.dot(weight))
    J = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))
    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])

##### PARTIAL DERIVATIVE WITH RESPECT TO WEIGHTS
def gradient(weight, X, y):
    m = y.size
    h = sigmoid(X.dot(weight.reshape(-1, 1)))
    grad = (1 / m) * X.T.dot(h - y)
    return (grad.flatten())


##### INITIALISING INITIAL WEIGHT AND CALCULATING INITIAL COST
initial_weight = np.zeros(X.shape[1])
cost = costFunction(initial_weight, X, y)
grad = gradient(initial_weight, X, y)
print('Cost: \n', cost)
print('Grad: \n', grad)
print('\n\n')

##### MINIMISING THE COST FUNCTION
res = minimize(costFunction, initial_weight, args=(X,y), method=None, jac=gradient, options={'maxiter':500})
print(res)

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))


# Assume Feature1 = 50 and Feature2=90.
# Predict using the optimized Theta values from above (res.x)
sigmoid(np.array([1, 50, 90]).dot(res.x.T))

p = predict(res.x, X)
print('Train accuracy {}%'.format(100*sum(p == y.ravel())/p.size))

print("\n\n DATA PLOT AFTER LOGISTIC REGRESSION\n")
print("=========================================\n\n")
plt.scatter(50, 90, s=60, c='k', marker='v', label='(50, 90)')
scatterPlot(data, 'Feature1', 'Feature2', 'Positive Class', 'Negative Class')
x1_min, x1_max = X[:,1].min(), X[:,1].max(),
x2_min, x2_max = X[:,2].min(), X[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');
plt.show()


#### REGULARIZATION
#### LOADING THE DATA
data2 = np.loadtxt('LogisticRegressionData2.txt', delimiter=',')
print('Dimensions: ',data.shape)
print(data[1:6,:])

#### INTIALISING FEATURE VECTOR AND LABEL VECTOR
y = np.c_[data2[:,2]]
X = data2[:,0:2]


print("\n\n DATA PLOT BEFORE LOGISTIC REGRESSION\n")
print("===========================================\n\n")
scatterPlot(data2, 'Feature1', 'Feature2', 'y = 1', 'y = 0')
plt.show()

# AUGMENTED VECTOR
poly = PolynomialFeatures(6)
XX = poly.fit_transform(data2[:,0:2])
XX.shape

# DEFINE COST FUNCTION
def costFunctionReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta))
    
    J = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (reg / (2 * m)) * np.sum(
        np.square(theta[1:]))

    if np.isnan(J[0]):
        return (np.inf)
    return (J[0])


# CALCULATE THE PARTIAL DERIVATIVE
def gradientReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1, 1)))

    grad = (1 / m) * XX.T.dot(h - y) + (reg / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]

    return (grad.flatten())


# INITIALISE THE WEIGHT AND CALCULATE THE INITIAL COST FUNCTION
initial_weight = np.zeros(XX.shape[1])
costFunctionReg(initial_weight, 1, XX, y)
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(17, 5))

# Decision boundaries
# Lambda = 0 : No regularization --> overfitting the training data
# Lambda = 1 : Moderate Penalty ---> Good Regularisation
# Lambda = 100 : High Penalty --> high bias ---> Even this is bad


print("\n\n DATA PLOT AFTER LOG. REG. + REGULARISATION\n")
print("==================================================\n\n")

for i, C in enumerate([0, 1, 100]):
    # Optimize costFunctionReg
    res2 = minimize(costFunctionReg, initial_weight, args=(C, XX, y), method=None, jac=gradientReg,
                    options={'maxiter': 3000})

    # Accuracy
    accuracy = 100 * sum(predict(res2.x, XX) == y.ravel()) / y.size

    # Scatter plot of X,y
    scatterPlot(data2, 'Feature1', 'Feature2', 'y = 1', 'y = 0', axes.flatten()[i])

    # Plot decisionboundary
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res2.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))

plt.show()