# Akane Sato
# ak700308

import numpy as np
import time
import scipy
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

'''
Dataset contains 214 instances.


Attribute Information (separated by commas):
1. Id number: 1 to 214 
2. RI: refractive index 
3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10) 
.
.
.
11. Type of glass: (class attribute) 
-- 1 building_windows_float_processed 
-- 2 building_windows_non_float_processed 
-- 3 vehicle_windows_float_processed 
-- 4 vehicle_windows_non_float_processed (none in this database) 
-- 5 containers 
-- 6 tableware 
-- 7 headlamps
'''

def main():

    data = open("../Data/glass.data", "r")
    X = []
    Y = []

    # Here, we are reading in the data from the file.
    # Array X contains the values of each characteristic.
    # Array Y contains the labels of each instance.
    for line in data:
        line = line.rstrip()
        line = line.split(',')

        newString = []

        for j in range (1, 10):
            newString.append(line[j])

        X.append(newString)
        Y.append(line[10])


    # Doing 5-fold cross validation, we will split data into 5 groups.
    # Each will contain 42 random instances (with four instances leftover).
    # 0 - 41 ; 42 - 83 ; 84 - 125 ; 126 - 167 ; 168 - 209 
    groups = np.random.permutation(214)


    # These contain each value that we will be testing for each hyperparameter.
    parameters = [ 
        {'kernel':['linear'], 'C':[0.01,0.1,0.5,1,2]},
        {'kernel':['rbf'], 'C':[0.01,0.1,0.5,1,2], 'gamma':[3,2,1,np.exp(-1),np.exp(-2)]},
        {'kernel':['sigmoid'], 'C':[0.01,0.1,0.5,1,2], 'gamma':[3,2,1,np.exp(-1),np.exp(-2)]},
        {'kernel':['poly'], 'C':[0.01,0.1,0.5,1,2], 'degree':[0,1,2,3], 'gamma':[3,2,1,np.exp(-1),np.exp(-2)]}
    ]

    x = 0

    while (x < 210):
        start = time.time()

        testingSet = []
        testingSetLabels = []
        trainingSet = []
        trainingSetLabels = []
        
        # For each iteration, one group of 42 data points will be the test.
        for i in range(x, x+42):
            testingSet.append(X[groups[i]])
            testingSetLabels.append(Y[groups[i]])

        # The remainder will be training set.
        for i in range(0, x):
            trainingSet.append(X[groups[i]])
            trainingSetLabels.append(Y[groups[i]])

        for i in range(x+42, 214):
            trainingSet.append(X[groups[i]])
            trainingSetLabels.append(Y[groups[i]])

        np.asarray(testingSet)
        np.asarray(testingSetLabels)
        np.asarray(trainingSet)
        np.asarray(trainingSetLabels)
        
        # decision_function_shape: default one-vs-rest (ovr), change to ovo for one-vs-one
        # remove class_weight='balanced' to test non-weighted classes
        # iid left as default 'false' to avoid command line warnings.
        clf = GridSearchCV(estimator=svm.SVC(decision_function_shape='ovo', class_weight='balanced'), param_grid=parameters, cv=5, iid='false', n_jobs=-1)
        clf.fit(trainingSet, trainingSetLabels)
        svm.SVC(C=clf.best_estimator_.C, gamma=clf.best_estimator_.gamma, kernel=clf.best_estimator_.kernel, degree=clf.best_estimator_.degree).fit(testingSet, testingSetLabels)


        '''
        print('Estimator score: ' + str(clf.best_score_) + '\nKernel: ' + clf.best_estimator_.kernel + '\nC: ' + str(clf.best_estimator_.C))
        if(clf.best_estimator_.kernel != 'linear'):
            print('Gamma: ' + str(clf.best_estimator_.gamma))
        if(clf.best_estimator_.kernel == 'poly'):
            print('Degree: ' + str(clf.best_estimator_.degree))


        print('Score on testing set: ' + str(clf.score(testingSet, testingSetLabels)))
        end = time.time()
        timetotal = end - start
        print('Time taken: ' + str(timetotal) + '\n')
        '''

        # Moves to the next group of 42 instances.
        x += 42



if __name__ == "__main__":
    main()