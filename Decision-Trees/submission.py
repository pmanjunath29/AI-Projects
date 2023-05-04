import numpy as np
import math
from collections import Counter
import time


class DecisionNode:
    """Class to represent a nodes or leaves in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """
        Create a decision node with eval function to select between left and right node
        NOTE In this representation 'True' values for a decision take us to the left.
        This is arbitrary, but testing relies on this implementation.
        Args:
            left (DecisionNode): left child node
            right (DecisionNode): right child node
            decision_function (func): evaluation function to decide left or right
            class_label (value): label for leaf node
        """
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Determine recursively the class of an input array by testing a value
           against a feature's attributes values based on the decision function.

        Args:
            feature: (numpy array(value)): input vector for sample.

        Returns:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice index for data labels.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data contained in the ReadMe.
    It must be built fully starting from the root.
    
    Returns:
        The root node of the decision tree.
    """
    func0 = lambda feature : feature[0] <= 0.06
    func1 = lambda feature : feature[2] <= -0.7
    func2 = lambda feature : feature[1] <= -1.5
    dt_root = DecisionNode(None, None, func0, None)
    dt_root.left = DecisionNode(None, None, None, 0)
    dt_root.right = DecisionNode(None, None, func1, None)
    dt_root.right.left = DecisionNode(None, None, None, 2)
    dt_root.right.right = DecisionNode(None, None, func2, None)
    dt_root.right.right.left = DecisionNode(None, None, None, 0)
    dt_root.right.right.right = DecisionNode(None, None, None, 1)
    return dt_root


def confusion_matrix(true_labels, classifier_output, n_classes=2):
    """Create a confusion matrix to measure classifier performance.
   
    Classifier output vs true labels, which is equal to:
    Predicted  vs  Actual Values.
    
    Output will sum multiclass performance in the example format:
    (Assume the labels are 0,1,2,...n)
                                     |Predicted|
                     
    |A|            0,            1,           2,       .....,      n
    |c|   0:  [[count(0,0),  count(0,1),  count(0,2),  .....,  count(0,n)],
    |t|   1:   [count(1,0),  count(1,1),  count(1,2),  .....,  count(1,n)],
    |u|   2:   [count(2,0),  count(2,1),  count(2,2),  .....,  count(2,n)],'
    |a|   .............,
    |l|   n:   [count(n,0),  count(n,1),  count(n,2),  .....,  count(n,n)]]
    
    'count' function is expressed as 'count(actual label, predicted label)'.
    
    For example, count (0,1) represents the total number of actual label 0 and the predicted label 1;
                 count (3,2) represents the total number of actual label 3 and the predicted label 2.           
    
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
    Returns:
        A two dimensional array representing the confusion matrix.
    """
    dict = {}
    for ind in range(len(true_labels)):
        if not true_labels[ind] in dict:
            dict[true_labels[ind]] = []
        dict[true_labels[ind]].append(ind)
    c_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cnt = 0
            if i in dict:
                for ind in dict[i]:
                    if classifier_output[ind] == j:
                        cnt += 1
            c_matrix[i][j] = cnt
    return c_matrix


def precision(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the precision of a classifier compared to the correct values.
    In this assignment, precision for label n can be calculated by the formula:
        precision (n) = number of correctly classified label n / number of all predicted label n 
                      = count (n,n) / (count(0, n) + count(1,n) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of precision of each classifier output. 
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [precision (0), precision(1), precision(2), ... precision(n)].
    """
    precisions = []
    if pe_matrix is None:
        pe_matrix = confusion_matrix(true_labels, classifier_output, n_classes)
    for c in range(n_classes):
        precisions.append(pe_matrix[c][c] / pe_matrix[:, c].sum())
    return precisions


def recall(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the recall of a classifier compared to the correct values.
    In this assignment, recall for label n can be calculated by the formula:
        recall (n) = number of correctly classified label n / number of all true label n 
                   = count (n,n) / (count(n, 0) + count(n,1) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of recall of each classifier output..
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [recall (0), recall (1), recall (2), ... recall (n)].
    """
    recalls = []
    if pe_matrix is None:
        pe_matrix = confusion_matrix(true_labels, classifier_output, n_classes)
    for c in range(n_classes):
        recalls.append(pe_matrix[c][c] / pe_matrix[c, :].sum())
    return recalls


def accuracy(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """Get the accuracy of a classifier compared to the correct values.
    Balanced Accuracy Weighted:
    -Balanced Accuracy: Sum of the ratios (accurate divided by sum of its row) divided by number of classes.
    -Balanced Accuracy Weighted: Balanced Accuracy with weighting added in the numerator and denominator.

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The accuracy of the classifier output.
    """
    if pe_matrix is None:
        pe_matrix = confusion_matrix(true_labels, classifier_output, n_classes)
    return np.trace(pe_matrix) / np.sum(pe_matrix)

def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0, 1, 2, ...
    Returns:
        Floating point number representing the gini impurity.
    """
    total = len(class_vector)
    counts = Counter(class_vector)
    gini_impurity = 0
    for key in counts:
        prob = counts[key] / total
        gini_impurity += (prob * (1 - prob))
    return gini_impurity

def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0, 1, 2....
        current_classes (list(list(int): A list of lists where each list has
            0, 1, 2, ... values).
    Returns:
        Floating point number representing the gini gain.
    """
    total_elements = 0
    for c in current_classes:
        total_elements += len(c)
    cur_gini_impurity = gini_impurity(previous_classes)
    for c in current_classes:
        cur_gini_impurity -= ((len(c) / total_elements) * (gini_impurity(c)))
    return cur_gini_impurity

class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=22):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        counts = Counter(classes)
        if depth >= self.depth_limit - 1:
            return DecisionNode(None, None, None, int(counts.most_common(1)[0][0]))
        elif len(counts) == 1:
            return DecisionNode(None, None, None, int(list(counts.keys())[0]))
        top_gain = 0
        top_avg = None
        index = None
        for i in range(len(features[0])):            
            col = features[:, i]
            avg = np.average(col)
            split_inds = (col <= avg)
            splits = [classes[split_inds], classes[~split_inds]]
            if len(splits[0]) >= 1 and len(splits[1]) >= 1:
                gain = gini_gain(classes, splits)
                if gain > top_gain:
                    top_gain = gain
                    top_avg = avg
                    index = i
        if index is None:
            return DecisionNode(None, None, None, int(counts.most_common(1)[0][0]))
        else:
            inds = features[:, index] <= top_avg
            func_not = lambda f: f[index] <= top_avg
            yes_inds =  self.__build_tree__(features[inds], classes[inds], depth + 1)
            not_inds =  self.__build_tree__(features[~inds], classes[~inds], depth + 1)
            return DecisionNode(yes_inds, not_inds, func_not, None)
    
    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        for feature in features:
            class_labels.append(self.root.decide(feature))
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    folds = []
    length = len(dataset[0])
    div_by_k = length // k
    all_inds = set(range(length))
    for i in range(k):
        inds = np.random.choice(length, div_by_k, False)
        train_inds = list(all_inds - set(inds))
        folds.append(((dataset[0][train_inds], dataset[1][train_inds]), (dataset[0][inds], dataset[1][inds])))
    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees=200, depth_limit=5, example_subsample_rate=.1,
                 attr_subsample_rate=.3):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        m = len(features)
        n = len(features[0])
        num_exs = int(m * self.example_subsample_rate)
        num_feats = int(n * self.attr_subsample_rate)
        for i in range(self.num_trees):
            inds = np.random.choice(m, num_exs, True)
            attrs = np.random.choice(n, num_feats, False)
            cur_tree = DecisionTree(self.depth_limit)
            cur_tree.fit(features[inds][:, attrs], classes[inds])
            self.trees.append((cur_tree, attrs))
        
    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        Returns:
            votes (list(int)): m votes for each element
        """
        votes = []
        for feature in features:
            classes = [tree.root.decide(feature[attrs]) for tree, attrs in self.trees]
            counts = Counter(classes)
            votes.append(int(counts.most_common(1)[0][0]))
        return votes


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, n_clf=0, depth_limit=0, example_subsample_rt=0.0, \
                 attr_subsample_rt=0.0, max_boost_cycles=0):
        """Create a boosting class which uses decision trees.
        Initialize and/or add whatever parameters you may need here.
        Args:
             num_clf (int): fixed number of classifiers.
             depth_limit (int): max depth limit of tree.
             attr_subsample_rate (float): percentage of attribute samples.
             example_subsample_rate (float): percentage of example samples.
        """
        self.num_clf = n_clf
        self.depth_limit = depth_limit
        self.example_subsample_rt = example_subsample_rt
        self.attr_subsample_rt=attr_subsample_rt
        self.max_boost_cycles = max_boost_cycles
        # TODO: finish this.
        raise NotImplemented()

    def fit(self, features, classes):
        """Build the boosting functions classifiers.
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        # TODO: finish this.
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features.
        Predict the labels for each feature in features to its corresponding class
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """
        # TODO: finish this.
        raise NotImplemented()


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """
        vectorized = np.add(np.square(data), data)
        return vectorized

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return (max_sum, max_sum_index)

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        # TODO: finish this.
        tmp = data[0:100, :].sum(axis = 1)
        return (np.max(tmp), np.argmax(tmp))
    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        unique_dict = {}
        flattened = data.flatten()
        for item in flattened:
            if item > 0:
                if item in unique_dict:
                    unique_dict[item] += 1
                else:
                    unique_dict[item] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        tmp = data[data > 0].flatten()
        return dict(Counter(tmp)).items()

    def non_vectorized_glue(self, data, vector, dimension='c'):
        """Element wise array arithmetic with loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            non_vectorized = np.ones((data.shape[0],data.shape[1]+1), dtype=float)
            non_vectorized[:, -1] *= vector
        elif dimension == 'r' and len(vector) == data.shape[1]:
            non_vectorized = np.ones((data.shape[0]+1,data.shape[1]), dtype=float)
            non_vectorized[-1, :] *= vector
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row, col] = data[row, col]
        return non_vectorized

    def vectorized_glue(self, data, vector, dimension='c'):
        """Array arithmetic without loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c':
            return np.column_stack((data, vector))
        return np.vstack((data, vector))
        # raise NotImplemented()
        #return vectorized

    def non_vectorized_mask(self, data, threshold):
        """Element wise array evaluation with loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared.
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        non_vectorized = np.zeros_like(data, dtype=float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val >= threshold:
                    non_vectorized[row, col] = val
                    continue
                non_vectorized[row, col] = val**2

        return non_vectorized

    def vectorized_mask(self, data, threshold):
        """Array evaluation without loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared. You are required to use a binary mask for this problem
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        vectorized = data.copy()
        vectorized[vectorized < threshold] = np.square(vectorized[vectorized < threshold])
        return vectorized


def return_your_name():
    # return your name
    # TODO: finish this
    #raise NotImplemented()
    return 'Pranav Shankar Manjunath'
