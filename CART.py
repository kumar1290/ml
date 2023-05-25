import numpy as np
import math 
from sklearn.datasets import load_iris

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.left = None
        self.right = None
        self.feature_index = 0
        self.threshold = 0


class Cart_Tree:
    def __init__(self, max_depth, acceptable_impurity):
        self.max_depth = max_depth
        self.acceptable_impurity = acceptable_impurity
   
    def predict(self, inputs):
        current_node = self.tree
        while current_node.left:
            if inputs[current_node.feature_index] < current_node.threshold:
                current_node = current_node.left
            else:
                current_node = current_node.right
        
        return current_node.predicted_class
    
    def fit(self, x, y):
        self.classifications = len(set(y))
        self.features = x.shape[1]
        self.tree = self.create_tree(x, y)
    
    def gini_impurity(y):
        instances = np.bincount(y)
        total = np.sum(instances)
        p = instances / total
        return 1.0 - np.sum(np.power(p, 2))
    
    
    def cart_split(self, x, y):
        m = y.size
        if m <= 1:
            return None, None
        best_index = None
        best_threshold = None
        parent = [np.sum(y == c) for c in range(self.classifications)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in parent)
    
        if best_gini >= self.acceptable_impurity:
            for index in range(self.features):
                thresholds, classes = zip(*sorted(zip(x[:, index], y)))
                num_left = [0] * self.classifications
                num_right = parent.copy()
                for i in range(1, m):
                    c = classes[i - 1]
                    num_left[c] += 1
                    num_right[c] -= 1
                    gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.classifications))
                    gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in
                    range(self.classifications))
                    gini = (i * gini_left + (m - i) * gini_right) / m
                    if thresholds[i] == thresholds[i - 1]:
                        continue
                    if gini < best_gini:
                        best_gini = gini
                        best_index = index
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2
            
        return best_index, best_threshold
    
    
    def create_tree(self, x, y, depth=0):
        samples_class = [np.sum(y == i) for i in range(self.classifications)]
        predicted_class = np.argmax(samples_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            index, thr = self.cart_split(x, y)
            if index is not None:
                indices_left = x[:, index] < thr
                x_left = x[indices_left]
                y_left = y[indices_left]
                x_right = x[~indices_left]
                y_right = y[~indices_left]
                node.feature_index = index
                node.threshold = thr
                node.left = self.create_tree(x_left, y_left, depth + 1)
                node.right = self.create_tree(x_right, y_right, depth + 1)
        
        return node


def evaluate(tree, test_data_m, label):
        tp,tn,fp,fn=0,0,0,0    
        for index in range(len(test_data_m)) :
            predOutput = tree.predict(test_data_m[index])  # predict the row
            actOutput = label[index]
            # print( actOutput," ",predOutput)
            if actOutput == 0:
                if predOutput== 0:
                    tp= tp+1
                else :
                    fn= fn+1
            else :
                if predOutput==0 :
                    fp= fp+1
                else :
                    tn= tn+1
        print("CF matr : ",tp,tn,fp,fn)
        print("Performance Analysis : ")
        accuracy = ((tp+tn)/(tp+fp+tn+fn))*100
        print("Accuracy : ",accuracy," %")
        precision = ((tp)/(tp+fp))*100
        print("Precision : ",precision," %")
        recall =((tp)/(tp+fn))*100
        print("Recall : ", recall," %")
        f1Score = (2*precision*recall)/(precision+recall)
        print("F1 Score : ", f1Score," %")
        # return accuracy

if __name__=="__main__":

    iris = load_iris()
    # print(iris.data[4],iris.target[4])
    # print(iris.data[:4])
    tot= len(iris.data)
    # print(type(iris.data))
    tree = Cart_Tree(max_depth=4, acceptable_impurity=0.2)
    tree.fit(iris.data[:math.floor(0.6*tot)], iris.target[:math.floor(0.6*tot)])
    # print(iris.data[4])
    cl = tree.predict(iris.data[4])
    print("Performance Evaluation : ")
    evaluate(tree,iris.data[math.floor(0.6*tot):],iris.target[math.floor(0.6*tot):])
    print('Classified as {}'.format(iris.target_names[cl]))