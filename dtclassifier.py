import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import operator

class DecisionTreeClassifier:
  def __init__(self):
    self.depth = 0
    self.features = list #two features name x1 x2
    self.X_train = np.array
    self.y_train = np.array
    self.num_feats = int
    self.train_size = int
  
  def fit(self, X, y):
      self.X_train = X
      self.y_train = y
      self.features = list(X.columns)
      self.train_size = X.shape[0]
      self.num_feats = X.shape[1]
      df = X.copy()
      df['y'] = y.copy()
      self.tree = self._build_tree(df)
      print("\nDecision Tree(depth = {}) : \n {}".format(self.depth, self.tree))


  def _build_tree(self, df, tree = None):
    #find feature with max information gain ratio
    feature, cutoff = self._find_best_split(df) #fxn to write

    if tree is None:
      tree = {}
      tree[feature] = {}
    # left 
    child_df = self._split_rows(df, feature, cutoff, operator.le)
    y, count = np.unique(child_df['y'], return_counts = True)

    self.depth += 1
    #same labels
    if (len(count) == 1):
      tree[feature]['<=' + str(cutoff)] = y[0]
    #diff labesl
    else:
      tree[feature]['<=' + str(cutoff)] = y[np.argmax(count)]

    #right
    child_df = self._split_rows(df, feature, cutoff, operator.gt)
    y, count = np.unique(child_df['y'], return_counts = True)

    #same labels
    if (len(count) == 1):
      tree[feature]['>' + str(cutoff)] = y[0]
    #diff labels
    else:
      tree[feature]['>' + str(cutoff)] = y[np.argmax(count)]

    return tree

  def _split_rows(self, df, feature, feat_value, operation):
    return df[operation(df[feature], feat_value)].reset_index(drop = True)

  def _find_best_split(self, df):
      #returns feature with max info gain ratio
    igr_list = []
    thresholds = []
    for feature in list(df.columns[:-1]):
      entropy_parent = self._get_entropy(df)
      entropy_feature_split, threshold = self._get_entropy_feature(df, feature)
      split_info = self._get_split_info(df, feature)

      info_gain_ratio = (entropy_parent - entropy_feature_split)/split_info
        
        #condition on info gain ration
       # if split_info != 0 and # all entropy are same condition; to do
      if split_info != 0:
        igr_list.append(info_gain_ratio)
        thresholds.append(threshold)

      #place holder till condition are written
    return df.columns[:-1][np.argmax(igr_list)], thresholds[np.argmax(igr_list)] 
    
  def _get_entropy(self, df):
    entropy=0
    for label in np.unique(df['y']):
      fraction = (df['y'].value_counts()[label])/(len(df['y']))
      entropy += - fraction * np.log2(fraction)

    return entropy

  def _get_entropy_feature(self, df, feature):
    entropy = 1
    threshold = None
    prev = 0

    for feat_value in np.unique(df[feature]):
      cur_entropy = 0
      cutoff = (feat_value + prev)/2

      for operation in [operator.le, operator.gt]:
        entropy_feature = 0
        for label in np.unique(df['y']):
          sd = len(df[feature][operation(df[feature], cutoff)][df['y'] == label])
          sk = len(df[feature][operation(df[feature], cutoff)])

          sdk = sd/sk
          
        entropy_feature += (-sdk) * np.log2(sdk)
        weighta = sk/len(df)
        cur_entropy += weighta * entropy_feature
        
      if cur_entropy < entropy:
        entropy, threshold = cur_entropy, cutoff
      prev = feat_value

    return entropy, threshold 
    
  def _get_split_info(self, df, feature):
    entropy = 1
    threshold = None
    prev = 0

    for feat_value in np.unique(df[feature]):
      cur_entropy = 0
      cutoff = (feat_value + prev)/2

      for operation in [operator.le, operator.gt]:
        entropy_feature = 0
        for label in np.unique(df['y']):
          sk = len(df[feature][operation(df[feature], cutoff)])
        weighta = sk/len(df)
        cur_entropy += weighta * np.log2(weighta)

    return cur_entropy

    def fit(self, X, y):
      self.X_train = X
      self.y_train = y
      self.features = list(X.columns)
      self.train_size = X.shape[0]
      self.num_feats = X.shape[1]
      df = X.copy()
      df['y'] = y.copy()
      self.tree = self._build_tree(df)
      print("\nDecision Tree(depth = {}) : \n {}".format(self.depth, self.tree))

    def _predict_target(self, feature_lookup, x, tree):

      for node in tree.keys():
        val = x[node]
        if type(val) == str:
          tree = tree[node][val]
        else:
          cutoff = str(list(tree[node].keys())[0]).split('<=')[1]

          if(val <= float(cutoff)):	#Left Child
            tree = tree[node]['<='+cutoff]
          else:						#Right Child
            tree = tree[node]['>'+cutoff]

        prediction = str

        if type(tree) is dict:
          prediction = self._predict_target(feature_lookup, x, tree)
        else:
          predicton = tree 
          return predicton

      return prediction   


    def predict(self, X):

      results = []
      feature_lookup = {key: i for i, key in enumerate(list(X.columns))}
		
      for index in range(len(X)):
        results.append(self._predict_target(feature_lookup, X.iloc[index], self.tree))

      return np.array(results)



datafile = '/content/drive/MyDrive/760/hw2/Druns.txt'
data = pd.read_table(datafile, sep = ' ', header=None, names = ['x1','x2','y1'])
                   
X, y = data.drop([data.columns[-1]], axis = 1), data[data.columns[-1]]
print(X)
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X, y)
#print(X,y)
print("\nTrain Accuracy: {}".format(accuracy_score(y, dt_clf.predict(X))))