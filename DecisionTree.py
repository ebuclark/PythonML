'''
Contains the classes DecisionTreeClassifier and DecisionTreeRegressor.
'''

import numpy as np

class Node():
    '''
    Auxiliary class to build the decision trees
    '''
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None

class DecisionTreeClassifier():
    def __init__(self, max_depth=100, min_samples_split=2, thresh=0.1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.thresh=thresh
        self.root = None

    def _split(self, X, thresh):
        '''
        Get all examples where x values are less than thresh and 
        greater than thresh
        '''
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx
    
    def _entropy(self, Y):
        '''
        Calculate the entropy for given classes
        '''
        proportions = np.bincount(Y)/len(Y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _information_gain(self, X, Y, thresh):
        '''
        Calculate the information gain of splitting data based on given threshold
        '''
        parent_loss = self._entropy(Y)
        left_idx, right_idx = self._split(X, thresh)

        n_left, n_right, n = len(left_idx), len(right_idx), len(Y)

        if (n_left == 0 or n_right == 0):
            return 0
        
        child_loss = (n_left / n) * self._entropy(Y[left_idx]) + \
            (n_right / n) * self._entropy(Y[right_idx])
        
        return parent_loss - child_loss

    def _optimal_split(self, X, Y, features):
        '''
        Find the optimal feature and threshold to split on based on largest
        information gain
        '''
        optimal_split = {
            'inf': -1,
            'feat': None,
            'thresh': None
        }

        for feat in features:
            X_feat = X[:,feat]
            threshold_vals = np.unique(X_feat)
            for thresh in threshold_vals:
                inf = self._information_gain(X_feat,Y, thresh)

                if (inf > optimal_split['inf']):
                    optimal_split['inf'] = inf
                    optimal_split['feat'] = feat
                    optimal_split['thresh'] = thresh
        return optimal_split['feat'], optimal_split['thresh']


    def _stop_criteria(self, depth):
        '''
        Determines when to stop growing the tree, returns a boolean specifying
        whether stopping criteria is met.
        '''
        if (depth >= self.max_depth or \
            self.n < self.min_samples_split or \
            self.n_classes == 1
        ):
            return True
        return False

    def _build_tree(self, X, Y, depth=0):
        '''
        A recursive function to build the tree using optimal splits
        '''
        self.n, self.p = X.shape
        self.n_classes = len(np.unique(Y))

        # Base case
        if self._stop_criteria(depth):
            value = np.argmax(np.bincount(Y))
            return Node(value=value)

        # Shuffle features
        random_features = np.random.choice(self.p, self.p, replace=False)
        # determine the optimal split
        opt_feat, opt_thresh = self._optimal_split(X,Y, random_features)

        # Split on optimal threshold value and recursively build tree for children
        left_idx, right_idx = self._split(X[:,opt_feat],opt_thresh)
        left_child = self._build_tree(X[left_idx,:], Y[left_idx], depth+1)
        right_child = self._build_tree(X[right_idx,:], Y[right_idx], depth+1)
        return Node(opt_feat, opt_thresh, left_child, right_child)

    def fit(self, X, Y):
        "Uses the CART training algorithm and entropy loss to fit the tree"
        if (type(Y) != np.ndarray or type(X) != np.ndarray):
            raise('Objects must be numpy arrays')
        
        self.root = self._build_tree(X,Y)

    def _traverse_tree(self, x, node):
        '''
        A function to recursively traverse the decision tree for a given x value
        '''
        # Base Case
        if node.is_leaf():
            return node.value

        # Recursive Case
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        '''
        Predicts for all examples the class by trversing the decision tree
        '''
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _do_printTree(self, node, tab=0):
        if (node.left is not None):
            print('  '*(tab+1), 'left')
            self._do_printTree(node.left, tab+1)

        print('  '*tab, node.feature, node.threshold, node.value)

        if (node.right is not None):
            print('  '*(tab+1), 'right')
            self._do_printTree(node.right, tab+1)


    def printTree(self):
        self._do_printTree(self.root, tab=0)


class DecisionRegressor():
    pass