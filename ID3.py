##############
# Name: Isha Mahadalkar
# Email: imahadal@purdue.edu
# PUID: 0030874031
# Date: 10/21/2020

# Number of Late Days used: 1

import numpy as np
import sys
import os
import pandas as pd
from math import log, pow
from pprint import pprint

# Defining the Classes 
# Each Node for the Tree
class Node:
    def __init__(self, left, right, attribute, threshold, label):
        self.left_subtree = left
        self.right_subtree = right
        self.attribute = attribute
        self.threshold = threshold
        self.label = label
        
# Super Class: Tree Class for our Decision Tree
class Tree:
    # Constructor for the Tree
    def __init__(self, training_file, test_file, model, root):
        self.trainingFile = training_file
        self.testFile = test_file
        self.model = model
        self.root = root
    
    # Function to calculate the number of instances for Survived from the data frame where x can be 0 or 1
    def count_Survived(self, data_frame, x):
        df_temp = data_frame.query('Survived == @x')
        return len(df_temp)
    
    # Function to calculate the number of instances for a given label with the given attr where x can be 0 or 1
    # and y can be any instance of the label or the threshold
    def count_survived_given_label(self, data_frame, label, x, y):
        query = "{} == @y".format(label)
        df_temp = data_frame.query(query)
        df_temp2 = df_temp.query('Survived == @x')
        return len(df_temp2)
    
    # Function to calculate the number of instances for a given label with the given attr where x can be 0 or 1
    # and y can be any instance of the label or the threshold
    # Returns less than and more than values 
    def count_survived_given_label_continuous(self, data_frame, label, x, y):
        query = "{} <= @y".format(label)
        df_temp = data_frame.query(query)
        df_temp2 = df_temp.query('Survived == @x')
        less = len(df_temp2)
        query = "{} > @y".format(label)
        df_temp = data_frame.query(query)
        df_temp2 = df_temp.query('Survived == @x')
        more = len(df_temp2)
        
        return less, more 
    
    # Splitting the data frame with the given value for the given attr
    # Returning the list of rows as groups which are being split at that threshold value
    def split(self, data_frame, value, attr):
        left_list = []
        right_list = []
        
        if attr == None or value == None:
          return left_list, right_list
        
        for index, row in data_frame.iterrows():
            if row[attr] <= value:
                left_list.append(row)
            else:
                right_list.append(row)
        
        return left_list, right_list
    
    # Finding the label as 'Survived' or 'Dead'
    def end_node_label(self, train_data):
        num_survived = self.count_Survived(train_data, 1)
        num_dead = self.count_Survived(train_data, 0)
        if num_survived >= num_dead:
            # return "Survived"  
            return 1     
        else:
            # return "Dead"
            return 0
        
                        
    # Function to calculate the entropy 
    # freqs - array of individual freqs    
    def entropy(self, freqs):
        all_freq = sum(freqs)
        entropy = 0
        for fq in freqs:
            if all_freq == 0:
                return 0.0
            prob = fq * 1.0 / all_freq
            if abs(prob) > 1e-8:
                entropy += -prob * np.log2(prob)
        return entropy
    
    # Function to calculate the Information Gain 
    # before_split_freqs - array of individual freqs before splitting on an attribute
    # after_split_freqs - array of array of individual freqs after splitting on the attribute
    # e.g input information_gain([9,5], [[2,2],[4,2],[3,1]])
    def information_gain(self, before_split_freqs, after_split_freqs):
        gain = self.entropy(before_split_freqs)
        overall_size = sum(before_split_freqs)
        weighted_sum = 0
        for freq in after_split_freqs:
            if overall_size == 0:
                return 0.0
            ratio = sum(freq) * 1.0 / overall_size
            gain -= ratio * self.entropy(freq)
        return gain
    
    # Function to make a prediction for a row from the testing data using the Tree created
    def make_prediction(self, root, row):
        row_t = pd.DataFrame(row).T
        # Setting the columns 
        row_t.columns = ["PClass", "Sex", "Age", "Fare", "Embarked", "Relatives", "isAlone"]
        
        if root.label != None:
            return root.label
        else:
            if row_t.iloc[0][root.attribute] <= root.threshold:
                return self.make_prediction(root.left_subtree, row)  
            else:
                return self.make_prediction(root.right_subtree, row)
    
    # Function to divide the training data set into k random data points  
    # Returns an array of dataframes with the k splits  
    def divide_k_fold(self, attr_data, label_data, k):
        fraction = float(1.0 / crossValidK)
        
        # Creating a data frame with the two files combined 
        train_data = pd.concat([attr_data, label_data], axis=1)
        
        shuffled_data = train_data.sample(frac=1)
        
        result = np.array_split(shuffled_data, k)
        
        return result
    
    # Function to split the training data frame 
    def split_data_frame(self, train_data):
        # Splitting the data frame     
        label_data = train_data[['Survived']].copy()
        label_data.columns = ['Survived']
        attr_data = train_data.copy()
        del attr_data['Survived']
        
        return attr_data, label_data
        
    
    # Main function to recursively build the Tree
    def ID3(self, attr_data, label_data, depth, max_depth, min_split):
        # Creating a data frame with the two files combined 
        train_data = pd.concat([attr_data, label_data], axis=1)
        # print(train_data)
        
        # Calculating the freqs before splitting 
        temp_f_bef = [self.count_Survived(label_data, 0), self.count_Survived(label_data, 1)]

        attr_split = None
        index = 0
        # Store inf_gain as a dict with the attr as the key 
        # and a list: [gain, threshold] as the value pair
        inf_gain = {}
        # Using a for loop to calculate the information gain of every attribute
        for att in attr_data:
            temp_f_aft = []
            # Finding the threshold values as a list and calculating the freqs
            val_list = train_data[att].unique().tolist()
            val_list.sort()
            t_list = []  
            inf_gain_list = {}             
               
            for i in range(len(val_list) - 1):
                t1 = float((float(val_list[i]) + float(val_list[i + 1])) / 2.0)
                t_list.append(t1)
            
            # Find the inf gain for each threshold
            if not t_list:
                continue
            
            for t in t_list:    
                temp_f_aft = []  
                l0 , m0 = self.count_survived_given_label_continuous(train_data, att, 0, t)
                l1 , m1 = self.count_survived_given_label_continuous(train_data, att, 1, t)
                temp_f_aft.append([l0, l1])
                temp_f_aft.append([m0, m1])
                inf_gain_list[t] = self.information_gain(temp_f_bef, temp_f_aft)

            threshold = max(inf_gain_list, key=inf_gain_list.get)
            # Add to the main inf gain list
            inf_gain[att] = [inf_gain_list[threshold], threshold]
                        
        # Picking the attribute with the maximum information gain to split on 
       
        if len(inf_gain) > 0:
            i = 1
            max_gain = 0
            for att in inf_gain:
                tl = inf_gain[att]
                if max_gain < tl[0]:
                    max_gain = tl[0]  
                    attr_split = att  
                    threshold = tl[1]          
                i = i + 1 
        else:
            max_gain = 0
            attr_split = None
            threshold = 0        
        
        # Building a node to hold the data;
        current_node = Node(None, None, attr_split, threshold, None)
        
        # Splitting the data into left and right parts
        left, right = self.split(train_data, threshold, attr_split)
        left = pd.DataFrame(left)
        right = pd.DataFrame(right)
        
        # Split the parent node into child nodes or stop the recursion and label the leaf nodes 
        # as 'survived' or 'dead'
    
        # 1. If theres no data samples in either of the left or right attr data 
        if left.empty or right.empty:
            current_node.label = self.end_node_label(train_data)
            return current_node
        
        # 2. If we have reached the max depth 
        if (depth >= max_depth):
            current_node.label = self.end_node_label(left)
            current_node.label = self.end_node_label(right)
            return current_node
        
        # Splitting the data to call ID3 again
        # Left Part
        left_part_label_data = left[['Survived']].copy()
        left_part_label_data.columns = ['Survived']
        left_part_attr_data = left.copy()
        del left_part_attr_data['Survived']
        
        # Right Part
        right_part_label_data = right[['Survived']].copy()
        right_part_label_data.columns = ['Survived']
        right_part_attr_data = right.copy()
        del right_part_attr_data['Survived']
             
        # 3. Calling ID3() for the left parts of the data
        # If the sample size in the left data is more than the minimum split data 
        # only then do we call the function again
        if len(left) > min_split:
            # Incrementing depth
            depth = depth + 1
            current_node.left_subtree = self.ID3(left_part_attr_data, left_part_label_data, depth, max_depth, min_split) 
        else:
            current_node.label = self.end_node_label(left)
            return current_node
            
        # 4. Calling ID3() for the right parts of the data
        # If the sample size in the right data is more than the minimum split data 
        # only then do we call the function again
        if len(right) > min_split:
            # Incrementing depth
            depth = depth + 1
            current_node.right_subtree = self.ID3(right_part_attr_data, right_part_label_data, depth, max_depth, min_split) 
        else:
            current_node.label = self.end_node_label(right)
            return current_node
        
        return current_node

# Function to read in the attribute data from the csv file       
def read_data(filename):
        file = open(filename)
        data_frame = pd.read_csv(file, delimiter= ',', na_values = 'NaN', index_col=None, engine='python')
        data_frame.columns = ["PClass", "Sex", "Age", "Fare", "Embarked", "Relatives", "isAlone"]
        
        attr_notnull = data_frame.dropna()
    
        # print(attr_notnull)
        mode_df = attr_notnull.mode()

        data_frame['PClass'].fillna(mode_df['PClass'][0], inplace = True)
        data_frame['Sex'].fillna(mode_df['Sex'][0], inplace = True)
        data_frame['Age'].fillna(mode_df['Age'][0], inplace = True)
        data_frame['Fare'].fillna(mode_df['Fare'][0], inplace = True)
        data_frame['Embarked'].fillna(mode_df['Embarked'][0], inplace = True)
        data_frame['Relatives'].fillna(mode_df['Relatives'][0], inplace = True)
        data_frame['isAlone'].fillna(mode_df['isAlone'][0], inplace = True)
        
        return data_frame

# Function to read in the label attribute data from the csv file       
def read_label_data(filename):
        file = open(filename)
        data_frame = pd.read_csv(file, delimiter= ',', index_col=None, engine='python')
        data_frame.columns = ["Survived"]
        return data_frame      

# Function to calculate the accuracy of the prediction
def calc_accuracy(test, pred):
    res = 0
    for i in range(len(test)):
        if test[i] == pred[i]:
            res = res + 1  
    return (res / float(len(test)))

# Method to print the Tree
def PrintTree(root):
    if root.left_subtree:
        PrintTree(root.left_subtree)
    pprint(vars(root)),
    if root.right_subtree:
        PrintTree(root.right_subtree)

# Method to print the k fold analysis 
def print_analysis(analysis):
    # Printing the analysis 
    length = len(analysis)
    for i in range(length):
        t_l = analysis[i]
        train_set_acc = t_l[0] * 100.0
        test_set_acc = t_l[1] * 100.0
        print("fold = " + str(i + 1) + ", training set accuracy = " + str(train_set_acc) + "%, validation set accuracy = " + str(test_set_acc)) + "%"
        

if __name__ == "__main__":
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser(description='CS373 Homework2 Decision Tree')
    parser.add_argument('--trainFolder')
    parser.add_argument('--testFolder')
    parser.add_argument('--model')
    parser.add_argument('--crossValidK', type=int, default=5)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--minSplit', type=int, default=2)
    args = parser.parse_args()
    

    # Arguments parsed from the command line
    # TODO: Change the training_file
    training_file = args.trainFolder + "/train-file.data"
    label_file = args.trainFolder + "/train-file.label"
    # training_file = "titanic-train.data"
    # label_file = "titanic-train.label"
    test_attr_file = args.testFolder + "/test-file.data"
    test_label_file = args.testFolder + "/test-file.label"
    model = args.model
    crossValidK = args.crossValidK
    if "depth" in args:
        max_depth = args.depth
    #max_depth = args.depth
    if "minSplit" in args:
        min_split = args.minSplit
    
    # Training Trees for K-Fold Validation 
    # analysis[k] = [train_acc, valid_acc]
    analysis = []
    trees = []
    
    # Building the decision tree
    # Implementing a binary decision tree with no pruning using the ID3
    if model.lower() == "vanilla":
        max_depth = float('inf')
        min_split = 0
    elif model.lower() == "depth":
        min_split = 0
    elif model.lower() == "minsplit":
        max_depth = float('inf')
    elif model.lower() == "postprune":
        max_depth = float('inf')
        min_split = 0
        
        
    # Creating a new Tree object       
    root = None
    new_tree = Tree(training_file, test_attr_file, model, root)

    # Reading the training data 
    attr_data = read_data(training_file)
    # Reading the label data 
    label_data = read_label_data(label_file)
    
    
    # Implementing K-Fold validation 
    # Dividing the training data for k-fold validation
    results = new_tree.divide_k_fold(attr_data, label_data, crossValidK)
    
    for i in range(crossValidK):
        temp_res = new_tree.divide_k_fold(attr_data, label_data, crossValidK)
        test_data = results[i]
        temp_res.pop(i)
        train_data = pd.concat(temp_res, ignore_index=True)
            
        # Finding the train and testing attr and label data
        test_attr, test_label = new_tree.split_data_frame(test_data)
        train_attr, train_label = new_tree.split_data_frame(train_data)
        # Calling the function to build the tree
        new_tree.root = new_tree.ID3(train_attr, train_label, 0, max_depth, min_split)       
        # PrintTree(new_tree.root)
        
        # Add the tree to the dictionary
        trees.append(new_tree)
        predictions_test = []
        predictions_train = []
        
        # Predict on testing set 
        for index, row in test_attr.iterrows():
            predictions_test.append(new_tree.make_prediction(new_tree.root, row))
            
        # Predict on training set & evaluate the training accuracy
        for index, row in train_attr.iterrows():
            predictions_train.append(new_tree.make_prediction(new_tree.root, row))
        
        
        train_set_acc = calc_accuracy(np.asarray(train_label), predictions_train)
        test_set_acc = calc_accuracy(np.asarray(test_label), predictions_test)
        
        analysis.append([train_set_acc, test_set_acc])
    
    # Printing the analysis
    print_analysis(analysis)
    
    # Running the model on the test set 
    # Reading the test data 
    test_attr_data = read_data(test_attr_file)
    test_label_data = read_label_data(test_label_file)
    
    total_predictions = []
    # Test each row of data for every Tree
            
    final_predictions = []
    for index, row in test_attr_data.iterrows():
        predictions_test = []
        for tree in trees:    
            predictions_test.append(tree.make_prediction(tree.root, row))    

        final_predictions.append(max(set(predictions_test), key=predictions_test.count))

    valid_acc = calc_accuracy(np.asarray(test_label_data), final_predictions) * 100.0
    print("Test set accuracy: " + str(valid_acc) + "%")