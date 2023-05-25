import pandas as pd
import numpy as np
import operator
from sklearn.datasets import load_iris


# making function for calculating euclidean distance
def EuclidianDistance(x1, x2, length):
    distance = 0
    for x in range(length):
        distance += np.square(x1[x] - x2[x])
    return np.sqrt(distance)

# making function for defining K-NN model
def knn(trainingSet, testInstance, k):
    distances = {}
    # print("test : ",testInstance)
    length = testInstance.shape[0]-1
    for x in range(len(trainingSet)):
        dist = EuclidianDistance(testInstance, trainingSet.iloc[x], length)
        # print("Dist : ",type(dist))
        distances[x] = dist
    sortdist = sorted(distances.items(), key=operator.itemgetter(1))
    neighbors = []
    # print("sort : ",sortdist)
    for x in range(k):
        neighbors.append(sortdist[x][0])
    # print("Negh  : ",neighbors)
    Count = {}  # to get most frequent class of rows
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x],-1]
        # print("res : ",response)
        if response in Count:
            Count[response] += 1
        else:
            Count[response] = 1
    sortcount = sorted(Count.items(), key=operator.itemgetter(1), reverse=True)
    return (sortcount[0][0], neighbors)


if __name__=="__main__":
    dataset = load_iris()
    dataset= pd.DataFrame(dataset.data)
    dataset['target']= load_iris().target
    dataset.head()
    [row,column] = dataset.shape
    no_of_neighbour = np.sqrt(row)
    k = (int)(no_of_neighbour)

    # supplying test data to the model
    print("Actual \t Predicted status ")
    corr=0
    for i in range(len(dataset)):
        result, neigh = knn(dataset, dataset.loc[i,:], k)
        print(dataset.iloc[i,-1],end="  ")
        print(result, end=" ")
        if dataset.iloc[i,-1]== result :
            print("Correct")
            corr+=1
        else:
            print("Wrong")
        # print(neigh)

    print("Number of correct output  : ",corr)