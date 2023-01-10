import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 0.001 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    self.conditional_prob_per_feat= None
    self.prior =None


  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    self.features = list(trainingData[0].keys()) # this could be useful for your code later...
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    ###########################################
    # Calculate the prior probabilities
    ###########################################
    frequency = util.Counter() #initialize a dict that counts the freq of each digit or each frew of T or false for faces
    for i in trainingLabels: frequency[i]+=1
    self.legalLabels=list(frequency.keys())
    self.prior=util.normalize(frequency)

    # the values a feature can take , in this case it will be 0 or 1 as our images are binary
    # for enhanced features it is a range [0..25] for face and [0..49] for digit
    m=0
    for i in range(len(trainingData)):
      x=max(trainingData[i].values())
      if(x>m): m=x
    #print("-------",m)
    feat_values=[i for i in range(m)]

    ###########################################
    # Set Occurances of all counters = 0
    # we will use maintain a Occurance counter that will 
    # count the number of occurance of 0 or 1 given the label assigned to the datum
    # OccuranceCounter = {(feat,label):
    #                           {0:number of occurances of 0 given label,
    #                            1:number of occurances of 1 given label}
    #                    } 
    ###########################################
    OccuranceCounter = {}
    for label in self.legalLabels:
      for feature in self.features:
        tempCounter = util.Counter()
        for _ in feat_values: tempCounter[_] = 0
        OccuranceCounter[(feature, label)] = tempCounter

    ###########################################
    # Segregate training data by label and 
    # Count the occurance of each feature
    ###########################################
    for i in range(len(trainingData)):
      featureValues = trainingData[i]
      label = trainingLabels[i]
      for feature in featureValues.keys():
        OccuranceCounter[(feature, label)][featureValues[feature]] += 1
    
    ###########################################
    # Smooth the occurances as 0 occurances can create a problem in multiplication further on
    ###########################################
    conditionalProbabilities_perfeature_perlabel = {}
    for label in self.legalLabels:
      for feature in self.features:
        CounterTuple = OccuranceCounter[feature, label]
        CounterTuple.incrementAll(CounterTuple.keys(), self.k)
        conditionalProbabilities_perfeature_perlabel[feature, label] = util.normalize(CounterTuple)
    self.conditionalProbabilities = conditionalProbabilities_perfeature_perlabel

#"*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    

    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    """
    logJoint = util.Counter()
    
    for label in self.legalLabels:
      logJoint[label] = math.log(self.prior[label])
      for feat, value in datum.items():
        phi = self.conditionalProbabilities[feat,label][value]     # Get the data we need from the sec dict
        if(phi<=0): phi=1.0000000001
        logJoint[label]= logJoint[label] + math.log( phi) # Calculate the joint probability 
        

      #logJoint[label]= probs

    #"*** YOUR CODE HERE ***"

    #util.raiseNotDefined()
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    """
    featuresOdds = []
        
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
