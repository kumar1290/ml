import pandas as pd
import numpy as np



# Entropy calculation for whole dataset
def calcTotalEntropy(train_data, label, class_list):
    total_row = train_data.shape[0]  # Total number of row in dataset
    total_entropy = 0
    for c in class_list:
        total_class_count = train_data[train_data[label] == c].shape[0]
        # No. of instances in each class i.e. YES or NO
        total_class_entr = - (total_class_count / total_row) * np.log2(total_class_count /
                                                                    total_row)  # Entropy formula
        total_entropy += total_class_entr
    return total_entropy


# For calculating entropy respect to another

def calc_entropy_of_class(feature_value_data, label, class_list):
    # Total no. of particular entity in an attribute
    class_count = feature_value_data.shape[0]
    entropy = 0
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
        entropy_class = 0
        if label_class_count != 0:
            probability_class = label_class_count / class_count
            entropy_class = - probability_class * np.log2(probability_class)
        entropy += entropy_class
    return entropy


def calc_info_gain(feature_name, train_data, label, class_list):
    # unqiue values of the feature
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0
    for feature_value in feature_value_list:
        feature_value_data = train_data[
        train_data[feature_name] == feature_value]  # filtering rows with that feature_value
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy_of_class(
        feature_value_data, label, class_list)
        feature_value_probability = feature_value_count / total_row
    # calculating information of the feature value
        feature_info += feature_value_probability * feature_value_entropy
    return calcTotalEntropy(train_data, label, class_list) - feature_info


def find_most_informative_feature(train_data, label, class_list):
    # finding the feature names in the dataset except target feature
    feature_list = train_data.columns.drop(label)
    max_info_gain = -1
    max_info_feature = None
    for feature in feature_list:
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
    # selecting feature name with highest information gain
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature

    return max_info_feature


def generate_sub_tree(feature_name, train_data, label, class_list):
    # dictionary of the count of unqiue feature value
    feature_value_count_dict = train_data[feature_name].value_counts(
        sort=False)
    tree = {}
    for feature_value, count in feature_value_count_dict.items():
        feature_value_data = train_data[
            train_data[feature_name] == feature_value]
        assigned_to_node = False  # flag for tracking feature_value is pure class or not
        for c in class_list:
            class_count = feature_value_data[feature_value_data[label] == c].shape[0]
            # count of feature_value = count of class (pure class)
            if class_count == count:
                tree[feature_value] = c  # adding node to the tree
 
    # removing rows with feature_value
            train_data = train_data[train_data[feature_name] != feature_value]
            assigned_to_node = True
        if not assigned_to_node:  # not pure class
            # node will further extend, so the branch is marked with ?
            tree[feature_value] = "?"
    
    return tree, train_data



def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0:  # if dataset becomes empty after updating
        max_info_feature = find_most_informative_feature(
            train_data, label, class_list)
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label,
                                            class_list)  # getting tree node and updated dataset
        next_root = None
        if prev_feature_value != None:  # add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else:  # add to root of the tree
            root[max_info_feature] = tree

        next_root = root[max_info_feature]
    for node, branch in list(next_root.items()):  # iterating the tree node
        if branch == "?":  # if it is expandable
        # using the updated dataset
            feature_value_data = train_data[train_data[max_info_feature] == node]
            make_tree(next_root, node, feature_value_data, label, class_list)

def predict(tree, instance):
    if not isinstance(tree, dict):  # if it is leaf node
        return tree
    else:
        # getting first key/feature name of the dictionary
        root_node = next(iter(tree))
        feature_value = instance[root_node]  # value of the feature
    # checking the feature value in current tree node
        if feature_value in tree[root_node]:
            # goto next feature
            return predict(tree[root_node][feature_value], instance)
        else:
            return None
        
def evaluate(tree, test_data_m, label):
    correct_preditct = 0
    wrong_preditct = 0
    tp,tn,fp,fn=0,0,0,0    
    for index, row in test_data_m.iterrows():
        result =predOutput = predict(tree, test_data_m.iloc[index])  # predict the row
        actOutput = test_data_m[label].iloc[index]
        if actOutput == "Yes":
            if predOutput=="Yes":
                tp= tp+1
            else :
                fn= fn+1
        else :
            if predOutput=="No":
                tn= tn+1
            else :
                fp= fp+1
        # predicted value and expected value is same or not
        if result == test_data_m[label].iloc[index]:
            correct_preditct += 1  # increase correct count
        else:
            wrong_preditct += 1  # increase incorrect count
    # accuracy = correct_preditct / (correct_preditct + wrong_preditct)
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


def calc_gain_ratio(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique() # unqiue values of the feature
    total_row = train_data.shape[0]
    feature_info = 0.0
    feature_split_info = 0.0
    for feature_value in feature_value_list:
    # filtering rows with that feature_value
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy_of_class(feature_value_data, label,class_list)
        # calculcating entropy for the feature value
        feature_value_probability = feature_value_count / total_row
        feature_value_split = - feature_value_probability * np.log2(feature_value_probability)
        feature_split_info += feature_value_split # calculating split info
        # calculating information of the feature value
        feature_info += feature_value_probability * feature_value_entropy
    Gain = calcTotalEntropy(train_data, label,class_list) - feature_info
    GainRatio = Gain / feature_split_info
    return GainRatio # calculating gain ratio


def find_most_informative_feature_c45(train_data, label, class_list):
    feature_list = train_data.columns.drop(label) # finding the feature names in the dataset
    max_gain_ratio = -1
    max_info_feature = None
    for feature in feature_list: # for each feature in the dataset
        feature_gain_ratio = calc_gain_ratio(feature, train_data, label, class_list)
        # selecting feature name with highest information gain
        if max_gain_ratio < feature_gain_ratio:
            max_gain_ratio = feature_gain_ratio
            max_info_feature = feature
    
    return max_info_feature

def make_tree_c45(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0:  # if dataset becomes empty after updating
        max_info_feature = find_most_informative_feature_c45(
            train_data, label, class_list)
        tree, train_data = generate_sub_tree(max_info_feature, train_data, label,
                                            class_list)  # getting tree node and updated dataset
        next_root = None
        if prev_feature_value != None:  # add to intermediate node of the tree
            root[prev_feature_value] = dict()
            root[prev_feature_value][max_info_feature] = tree
            next_root = root[prev_feature_value][max_info_feature]
        else:  # add to root of the tree
            root[max_info_feature] = tree

        next_root = root[max_info_feature]
    for node, branch in list(next_root.items()):  # iterating the tree node
        if branch == "?":  # if it is expandable
        # using the updated dataset
            feature_value_data = train_data[train_data[max_info_feature] == node]
            make_tree_c45(next_root, node, feature_value_data, label, class_list)


def id3(trainData, label):
    train_data = trainData.copy()
    tree = {}  # tree which will be updated
    # getting unqiue classes of the label
    class_list = train_data[label].unique()
    make_tree(tree, None, trainData, label, class_list)
    return tree

def c4point5(trainData, label):
    train_data = trainData.copy()
    tree = {} #tree which will be updated
    class_list = train_data[label].unique() #getting unqiue classes of the label
    make_tree_c45(tree, None, trainData, label, class_list) #start calling recursion
    return tree


if __name__=="__main__":
    data = pd.read_csv('ML\datasets\\tennis.csv')
    print("Data : ")
    print(data)
    id3Tree = id3(data, 'PlayTennis') 
    test_data_m = data.loc[:200,:]
    print("\n\nPerformance Matrix ID3 ")
    evaluate(id3Tree, test_data_m, 'PlayTennis')
    c4Tree = c4point5(data, 'PlayTennis')
    print("\n\nPerformance Matrix c4.5 ")
    evaluate(c4Tree, test_data_m, 'PlayTennis')
    # trainData = trainData.loc[:200, :]