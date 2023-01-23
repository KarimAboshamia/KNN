import pandas as pd

# Prepare data
features = pd.read_csv("BankNote_Authentication.csv")                   #Read
tiebreaker = features['class'].iloc[0]                                  #The tiebreaker is the class that comes first in the Train file
features = features.sample(frac=1)                                      #Shuffle data
result = features['class']                                              #Extract result class
resultArray = [result.iloc[i] for i in range(len(result.axes[0]))]      #Regular array version of results
features = features.drop(features.columns[4], axis=1)                   #Drop result from features set
# features = features.drop(features.columns[3], axis=1)                 #Drop entropy from features set as it is not an indicator
# features = features.drop(features.columns[0], axis=1)                 #Drop
numFeatures = len(features.axes[1])                                     #Get number of features (Columns)
numRows = len(features.axes[0])                                         #Get number of entries  (Rows)

# Select best features
# for f in features:
#     plt.scatter(result,features[f])
#     plt.ylabel(str(f))
#     plt.xlabel("class")
#     plt.show()

# Normalize features
#features = (features-features.mean()) / features.std()

# Get train and test data
trainPercentage = 0.7
features_train = features.iloc[:int(trainPercentage*numRows)]           #Create the list of features used in training
features_train = (features_train - features_train.mean()) / features_train.std()
result_train   = resultArray[:int(trainPercentage*numRows)]             #Create the list of results  used in training

features_test  = features.iloc[int(trainPercentage*numRows):]           #Create the list of features used in testing
features_test = (features_test - features_test.mean()) / features_test.std()
#We can't normalize dataset using the mean & std of a different dataset

result_test    = resultArray[int(trainPercentage*numRows):]             #Create the list of results  used in testing
assert (len(result_test) + len(result_train) - numRows) == 0            #Test integrity


# Start of KNN model #######################################
def getSquaredEuclideanDistance(point1, point2):                  #Helper function to get distance between 2 points
    delta = point2 - point1                                       #Vector subtraction of points feature values
    deltaSquare = pow(delta,2)                                    #Vector of delta values squared
    summation = 0                                                 #float: Summation of values in deltaSquared
    for i in range(len(deltaSquare)):                             #Loop to sum values from deltaSquare ,equivalent
        summation += (float(deltaSquare.iloc[i]))                               #to #np.sum(deltaSquare)

    # return sqrt(summation)
    return summation

def getKNearestPoints(allPoints,classes,point,k):                           #Get the nearest K points from a target
    distances = []                                                             #A dictionary [key = originalIndex, val = distance]
    numEntries = len(allPoints.axes[0])                                         #Get number of points
    for i in range(numEntries):                                                 #For every index (point) in allPoints
        p = allPoints.iloc[i]                                                   #Get the point in this index
        distances.append([getSquaredEuclideanDistance(p, point),classes[i]])       #Add entry in distances with calculated distance
    distances.sort(key=lambda item: item[0])                   #Sort distances{} according to value (distance)
    nearestK = []
    for i in range(1,k+1):                                                      #Extract the first k entries
        if i >= numEntries:                                                     #If points fewer than k, stop
            break
        nearestK.append([distances[i][0],distances[i][1]])

    return nearestK

def classify(nearstK):                                    #Get the most common class in a set of points
    ones = 0                                                        #Counter of class 1 instances
    zero = 0                                                        #Counter of class 0 instances

    for i in range(len(nearstK)):                                   #for every entry in form of [index, class]
        if nearstK[i][1] == 0:                                      #add to the corresponding counter
            zero += 1
        else:
            ones += 1

    if ones == zero:
        return tiebreaker
    else:
        return int(ones > zero)


def runKnnModel(testPoints, trainPoints, trainClasses, k, printPercentage = False):                   #Interface function to run the model
    predictions = []                                                            #List of predictions
    myPoints = trainPoints                                                      #List of processed point
    numEntries = len(testPoints.axes[0])                                          #Number of data items
    for i in range(numEntries):                                                 #For every point in the original points
        newPoint = testPoints.iloc[i]                                             #Get point
        nearstK = getKNearestPoints(myPoints, trainClasses+predictions, newPoint, k)         #Get nearest k points
        myClass = classify(nearstK)                                             #Classify according to the nearest K

        myPoints = pd.concat([myPoints,newPoint.to_frame().T])                  #Add the current point  to the list of processed points equivalent to # myPoints.append(newPoint)
        if printPercentage:                                                     #Print percentage
            print("Working: ",round(float(i)/numEntries*100)," %")
        predictions.append(myClass)                                             #Add the new prediction to the list of predictions
    return predictions


def reportAccuracy(results, predictions, k):
    numEntries = len(predictions)
    correct = 0
    wrong = 0
    for i in range(numEntries):
        if results[i] == predictions[i]:
            correct += 1
        else:
            wrong += 1
    accuracy = correct/numEntries

    print("K value: ", k)
    print("Number of correctly classified instances: ", correct, " Total number of instances: ", numEntries)
    print("Accuracy: ",accuracy)

# End of KNN Model #########################################


# Main()
for k in [2,3,5]:
    predictions = runKnnModel(features_test,features_train, result_train,k, printPercentage=False)
    print(predictions)
    reportAccuracy(result_test, predictions, k)
    print("#########################\n")
# getKNearestPoints(features_train,result_train,point1,k)