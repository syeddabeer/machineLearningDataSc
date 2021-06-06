##################################### PLA ALGORITHM - ON RAW DATA WITH 256 FEATURES AND ALSO MAKING PREDICTIONS ON TEST RAW DATA ##################################3
import numpy as np
from random import choice
from numpy import array, dot, random
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# Initially, the data is read from internet as given below, written to a csv file in local machine as given below, to decrease dependency of this code on internet
# train = pd.read_csv('http://amlbook.com/data/zip/zip.train', delimiter=' ', header=None)
# train.to_csv('train.csv', index=False)

# Load CSV file from the local machine
train = pd.read_csv('C:/Users/syedd/train.csv', delimiter=',', header='infer')
train.shape
del train['257']
train.shape

print("WORKING WITH THE RAW DATA THAT HAS 256 FEATURES CONTAINING GREY SCALE VALUES FOR 256 PIXELS")
print("============================================================================================ \n \n")
print("PLA ALGORITHM")
print("=============\n \n")

print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
# Store the labels in a vector. Here trainlabels is 'y' in our formula
trainlabels = train['0']
#Labels should be integer, not float numbers
trainlabels=trainlabels.astype(int)

del train['0']
# The data should be represented in matrix form
train.shape
traindata = np.asmatrix(train.loc[:,:])
traindata.shape

# Visualising a digit using the 256 grey scale values of the 16 by 16 pixels
#Taking a sample row
samplerow = traindata[21:22]
#reshape it to 16*16 grid
samplerow = np.reshape(samplerow,(16,16))
print("A sample digit from the dataset:")
plt.imshow(samplerow, cmap="hot")
plt.show()

# Initialize the weight matrix
weights = np.zeros((10,256))
print(weights.shape)


################ PLA ALGORITHM##############################
E = 15
errors = []
for epoch in range(E):
    err = 0 #reset the error
    # For each handwritten digit in training set,
    for i, x in enumerate(traindata):
        dp=[] #create a new container for the calculated dot products
        # For each digit class (0-9)
        for w in weights:
            dotproduct = np.dot(x,w)
            #take the dot product of the weight and the data
            dp.append(dotproduct)
            #add the dot product to the list of dot products
        guess = np.argmax(dp)
        #take the largest dot product and make that the guessed digit class
        actual = trainlabels[i]
        # If the guess was wrong, update the weight vectors
        if guess != actual:
            weights[guess] = weights[guess] - x #update the incorrect (guessed) weight vector
            weights[actual] = weights[actual] + x #update the correct weight vector
            err += 1
    errors.append(err/7291) #track the error after each pass through the training set
plt.plot(list(range(0,E)),errors) #plot the error after all training epochs
plt.title('Error Rate while using PLA Algorithm')
plt.show()


################################################ TEST DATA ###############################################
##Reading from internet
# test  = pd.read_csv('http://amlbook.com/data/zip/zip.test', delimiter=' ', header=None)
# test.to_csv('test.csv', index=False)

test = pd.read_csv('C:/Users/syedd/test.csv', delimiter=',', header='infer')

test.shape
del test['257']
test.shape

print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
testlabels = test['0']
testlabels=testlabels.astype(int)
actualvalues=test['0']
del test['0']
test.shape
testdata = np.asmatrix(test.loc[:,:])
testdata.shape

guesses = []
for i, z in enumerate(testdata):
    dp=[]
    for w in weights:
        dotproduct = np.dot(z,w)
        dp.append(dotproduct)
    guess = np.argmax(dp)
    guesses.append(guess)

print("Preparing the Predictions file with the guesses")
Prediction = pd.DataFrame(data= {'Prediction': guesses})
Prediction.shape
Prediction.to_csv('predictions_On_Test_Data_Using_PLA_Algorithm.csv', index=False)


from sklearn.metrics import mean_squared_error
from math import sqrt
print("Error Rate for the PLA Algorithm: {}".format(sqrt(mean_squared_error(testlabels, guesses))))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(testlabels, guesses)
print("Accuracy for the PLA Algorithm:")
print(accuracy)
print("\n")


##################################### POCKET PLA ALGORITHM - ON RAW DATA WITH 256 FEATURES AND ALSO MAKING PREDICTIONS ON TEST RAW DATA ##################################3

import numpy as np
from random import choice
from numpy import array, dot, random
import matplotlib.pyplot as plt
import pandas as pd

#Reading from Internet
# train = pd.read_csv('http://amlbook.com/data/zip/zip.train', delimiter=' ', header=None)
# train.to_csv('train.csv', index=False)

train = pd.read_csv('C:/Users/syedd/train.csv', delimiter=',', header='infer')
train.shape
del train['257']
train.shape

print("POCKET PLA ALGORITHM")
print("====================\n \n")

print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
trainlabels = train['0']
trainlabels=trainlabels.astype(int)

del train['0']
train.shape
traindata = np.asmatrix(train.loc[:,:])
traindata.shape

samplerow = traindata[16:17]
samplerow = np.reshape(samplerow,(16,16))
print("A sample digit from the dataset:")
plt.imshow(samplerow, cmap="hot")
plt.show()

weights = np.zeros((10,256))
print(weights.shape)

################POCKET PLA ALGORITHM##############################
max_iters = 32
N = len(traindata)
best = None
best_miscount = N + 1
success = False
iters = 0
indexes = range(N)
while iters < max_iters:
    num_misidentified = 0
    fix_index = -1
    for i, x in enumerate(traindata):
        dp=[]
        for w in weights:
            dotproduct = np.dot(x,w)
            dp.append(dotproduct)
        guess = np.argmax(dp)
        actual = trainlabels[i]
        if guess != actual:
            num_misidentified += 1
            weights[guess] = weights[guess] - x
            weights[actual] = weights[actual] + x
            if fix_index < 0:
                fix_index = i
    if num_misidentified < best_miscount:
        best = weights.copy()
        best_miscount = num_misidentified

    if num_misidentified == 0:
        exit()

    iters += 1


################################################ TEST DATA ###############################################
##Reading from internet
# test  = pd.read_csv('http://amlbook.com/data/zip/zip.test', delimiter=' ', header=None)
# test.to_csv('test.csv', index=False)

test = pd.read_csv('C:/Users/syedd/test.csv', delimiter=',', header='infer')
test.shape
del test['257']
test.shape


print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
testlabels = test['0']
testlabels=testlabels.astype(int)
actualvalues=test['0']
del test['0']
test.shape
testdata = np.asmatrix(test.loc[:,:])
testdata.shape

guesses = []
for i, z in enumerate(testdata):
    dp=[]
    for w in weights:
        dotproduct = np.dot(z,w)
        dp.append(dotproduct)
    guess = np.argmax(dp)
    guesses.append(guess)

print("Preparing the Predictions file with the guesses \n")
Prediction = pd.DataFrame(data= {'Prediction': guesses})
Prediction.shape
Prediction.to_csv('predictions_On_Test_Data_Using_Pocket_PLA_Algorithm.csv', index=False)

import sklearn
from sklearn.metrics import mean_squared_error
from math import sqrt
print("Error Rate for the Pocket PLA Algorithm: {}".format(sqrt(mean_squared_error(testlabels, guesses))))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(testlabels, guesses)
print("Accuracy for the Pocket PLA Algorithm:")
print(accuracy)
print("\n")


##################################### PLA AND POCKET PLA ALGORITHM - ON PROCESSED DATA WITH 2 FEATURES AND DRAWING GRAPHS SHOWING WEIGHT LINE ##################################3

import numpy as np
from random import choice
from numpy import array, dot, random
import matplotlib.pyplot as plt
import pandas as pd

# trainProcessed = pd.read_csv('http://amlbook.com/data/zip/features.train', delimiter=' ', usecols=[3, 6, 8], header=None)
# trainProcessed.to_csv('trainProcessed.csv', index=False)

# trainProcessed = pd.read_csv('C:/Users/syedd/trainProcessed.csv', delimiter=',', header='infer', nrows=100)
trainProcessed = pd.read_csv('C:/Users/syedd/trainProcessed.csv', delimiter=',', header='infer')
trainProcessed.shape

print("WORKING WITH THE PROCESSED DATA THAT HAS ONLY TWO FEATURES INTENSITY AND SYMMETRY")
print("================================================================================= \n \n")

print("Training set has {0[0]} rows and {0[1]} columns".format(trainProcessed.shape))
print("\n")

y = trainProcessed['3']
del trainProcessed['3']
data = trainProcessed

y=y.astype(int)

y[y==0] = -1
y[y==2] = -1
y[y==3] = -1
y[y==4] = -1
y[y==5] = -1
y[y==6] = -1
y[y==7] = -1
y[y==8] = -1
y[y==9] = -1

"To calculate wx-threshold. weight vector is unknown. Here x is just x. -threshold is bias. if you add bias, it becomes WX where X is (1,x)"
def sign(data):
    c = np.asarray(np.sign(data), dtype=int)
    c[c==0] = 1
    return c

def plot_costs(x0, x1, y):
    for i, c in enumerate(y):
        plt.scatter(x0[i], x1[i], marker='+' if c==1 else '$-$', color='b' if c==1 else 'r', s=50)

def add_one_column(data):
    N = len(data) # number of records
    return np.c_[np.ones(N), data] # add column of ones for x_0

def plot_weight_line(weights, x0, x1):
    def eq(w, x):
        """ convert w0 + w1*x + w2*y into y = mx + b"""
        return (-w[1]*x - w[0]) / w[2]
    plt.plot([x0, x1], [eq(weights, x0), eq(weights, x1)], ls='--', color='g')

def plot_weight_example(weights):
    plot_weight_line(weights, 0, 0.75)
    plot_costs(data[:, 0], data[:, 1], y)
    plt.xlim(0, 1);
    plt.ylim(-8, 1);
    plt.show()


def PLA(xs, y, weights=None, max_iters=5000):
    if weights is None:
        weights = np.array([np.random.random(xs.shape[1])])
    if weights.ndim == 1:
        weights = np.array([weights])
    misidentified = True
    success = False
    iters = 0
    indexes = range(len(xs))
    while misidentified and iters < max_iters:
        misidentified = False
        for i in np.random.permutation(indexes):
            x = xs[i]
            s = sign(np.dot(weights, x)[0])
            if s != y[i]:
                misidentified = True
                weights += np.dot(y[i], x)
                break
        success = not misidentified
        iters += 1

    return weights, success, iters


def PPLA(xs, y, weights=None, max_iters=5200):
    N = len(xs)
    if weights is None:
        weights = np.array([np.random.random(xs.shape[1])])
    if weights.ndim == 1:
        weights = np.array([weights])
    best = None
    best_miscount = N + 1
    success = False
    iters = 0
    indexes = range(N)
    while iters < max_iters:
        num_misidentified = 0
        fix_index = -1
        for i in np.random.permutation(indexes):
            x = xs[i]
            s = sign(np.dot(weights, x)[0])
            if s != y[i]:
                num_misidentified += 1
                if fix_index < 0:
                    fix_index = i
        if num_misidentified < best_miscount:
            best = weights.copy()
            best_miscount = num_misidentified

        if num_misidentified == 0:
            return weights, True, iters, 0

        weights += np.dot(y[fix_index], xs[fix_index])
        iters += 1

    return best, False, iters, best_miscount

print("\n \n \t \t GRAPHS DRAWN USING THE PLA ALGORITHM")
print("\t \t===================================== \n \n")

from numpy.random import randn
d = 2
xs = add_one_column(data)
weights = random.rand(3)

plot_weight_line(weights, 0, 0.75)
plot_costs(xs[:, 1], xs[:, 2], y)
plt.title('PLA Algorithm Starts')
plt.show()

# run algorithm
weights, success, iters = PLA(xs, y, weights)

# plot and print the results
plt.figure()
plot_costs(xs[:, 1], xs[:, 2], y)
print('final weights', weights)
plot_weight_line(weights[0, :], 0, 0.75)
plt.title('Graph after the PLA Algorithm Result')
plt.show()
print('numer of iterations', iters)


#########################################Pocket PLA####################3


ns_data=xs
ns_y=y
ns_xs = ns_data

print("\n \n \t \t GRAPHS DRAWN USING THE POCKET PLA ALGORITHM")
print("\t \t============================================= \n \n")


ns_weights, success, iters, num_errors = PPLA(ns_xs, ns_y, max_iters=5000)
plot_costs(ns_xs[:, 1], ns_xs[:, 2], ns_y)
plot_weight_line(ns_weights[0, :], 0,0.75)
plt.show()

import scipy.linalg as la
xi = la.pinv(ns_xs)
w_lr = np.dot(xi, ns_y)
ns_weights, success, iters, num_errors = PPLA(ns_xs, ns_y, w_lr, max_iters=5000)

plot_costs(ns_xs[:, 1], ns_xs[:, 2], ns_y)
plot_weight_line(w_lr, 0, 0.75)
plt.show()
print(w_lr)

w, _, _, _ = np.linalg.lstsq(ns_xs, ns_y)
print(w)
print(w_lr)


plot_costs(ns_xs[:, 1], ns_xs[:, 2], ns_y)
plot_weight_line(w_lr, 0, 0.75)
plt.show()

datas=np.asarray(data.loc[:,:])
x2 = add_one_column(datas*datas)

plot_costs(x2[:, 1], x2[:, 2], y)
weights2, success, iters, num_errors = PPLA(x2, y, max_iters=5000)
plot_weight_line(weights2[0], 0, 0.75)
plt.title('Graph after the Pocket PLA Algorithm Result')
plt.show()
