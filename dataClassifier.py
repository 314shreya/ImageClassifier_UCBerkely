# This file contains feature extraction methods and harness 
# code for data classification

import naiveBayes
import knnClassifier
import knnClassifier_Digits
import knnClassifier_Faces
import perceptron
import minicontest
import samples
import sys
import util
import time
import numpy as np
import random

TEST_SET_SIZE = 100
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_WIDTH=60
FACE_DATUM_HEIGHT=70


def basicFeatureExtractorDigit(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(DIGIT_DATUM_WIDTH):
    for y in range(DIGIT_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def basicFeatureExtractorFace(datum):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is an edge (1) or no edge (0)
  """
  a = datum.getPixels()

  features = util.Counter()
  for x in range(FACE_DATUM_WIDTH):
    for y in range(FACE_DATUM_HEIGHT):
      if datum.getPixel(x, y) > 0:
        features[(x,y)] = 1
      else:
        features[(x,y)] = 0
  return features

def enhancedFeatureExtractorDigit(datum):
  """
  Your feature extraction playground.
  
  You should return a util.counter() of features
  for this datum (datum is of type samples.Datum).
  
  ## DESCRIBE YOUR ENHANCED FEATURES HERE...
  
  ##
  """
  block_size = 4
  features = util.Counter()
  for x in range(0, DIGIT_DATUM_WIDTH, block_size):
    for y in range(0, DIGIT_DATUM_HEIGHT, block_size):
      for x1 in range(block_size):
        for y1 in range(block_size):
          if datum.getPixel(x+x1, y+y1) > 0:
            features[(x,y)] += 1 # counting the number of pixels in 7x7 grid
          else:
            features[(x,y)] += 0
  return features

  
  "*** YOUR CODE HERE ***"
  
  return features


def contestFeatureExtractorDigit(datum):
  """
  Specify features to use for the minicontest
  """
  features =  basicFeatureExtractorDigit(datum)
  return features

def enhancedFeatureExtractorFace(datum):
  """
  Your feature extraction playground for faces.
  It is your choice to modify this.
  """
  block_size = 5
  features = util.Counter()
  for x in range(0, FACE_DATUM_WIDTH, block_size):
    for y in range(0, FACE_DATUM_HEIGHT, block_size):
      for x1 in range(block_size):
        for y1 in range(block_size):
          if datum.getPixel(x+x1, y+y1) > 0:
            features[(x,y)] += 1 # counting the number of pixels in 5x5 grid
          else:
            features[(x,y)] += 0
  return features

def analysis(classifier, guesses, testLabels, testData, rawTestData, printImage):
  """
  This function is called after learning.
  Include any code that you want here to help you analyze your results.
  
  Use the printImage(<list of pixels>) function to visualize features.
  
  An example of use has been given to you.
  
  - classifier is the trained classifier
  - guesses is the list of labels predicted by your classifier on the test set
  - testLabels is the list of true labels
  - testData is the list of training datapoints (as util.Counter of features)
  - rawTestData is the list of training datapoints (as samples.Datum)
  - printImage is a method to visualize the features 
  (see its use in the odds ratio part in runClassifier method)
  
  This code won't be evaluated. It is for your own optional use
  (and you can modify the signature if you want).
  """
  
  # Put any code here...
  # Example of use:
  for i in range(len(guesses)):
      prediction = guesses[i]
      truth = testLabels[i]
      if (prediction != truth):
          print("===================================")
          print("Mistake on example %d" % i) 
          print("Predicted %d; truth is %d" % (prediction, truth))
          print("Image: ")
          #print(rawTestData[i])
          break


## =====================
## You don't have to modify any code below.
## =====================


class ImagePrinter:
    def __init__(self, width, height):
      self.width = width
      self.height = height

    def printImage(self, pixels):
      """
      Prints a Datum object that contains all pixels in the 
      provided list of pixels.  This will serve as a helper function
      to the analysis function you write.
      
      Pixels should take the form 
      [(2,2), (2, 3), ...] 
      where each tuple represents a pixel.
      """
      image = samples.Datum(None,self.width,self.height)
      for pix in pixels:
        try:
            # This is so that new features that you could define which 
            # which are not of the form of (x,y) will not break
            # this image printer...
            x,y = pix
            image.pixels[x][y] = 2
        except:
            print("new features:", pix)
            continue
      print(image)  

def default(str):
  return str + ' [Default: %default]'

def readCommand( argv ):
  "Processes the command used to run from the command line."
  from optparse import OptionParser  
  parser = OptionParser(USAGE_STRING)
  
  parser.add_option('-c', '--classifier', help=default('The type of classifier'), choices=['nb', 'naiveBayes', 'perceptron', 'minicontest', 'knnClassifier'], default='naiveBayes')
  parser.add_option('-d', '--data', help=default('Dataset to use'), choices=['digits', 'faces'], default='digits')
  parser.add_option('-t', '--training', help=default('The size of the training set'), default=100, type="int")
  parser.add_option('-f', '--features', help=default('Whether to use enhanced features'), default=False, action="store_true")
  parser.add_option('-o', '--odds', help=default('Whether to compute odds ratios'), default=False, action="store_true")
  parser.add_option('-1', '--label1', help=default("First label in an odds ratio comparison"), default=0, type="int")
  parser.add_option('-2', '--label2', help=default("Second label in an odds ratio comparison"), default=1, type="int")
  parser.add_option('-k', '--smoothing', help=default("Smoothing parameter (ignored when using --autotune)"), type="float", default=2.0)
  parser.add_option('-a', '--autotune', help=default("Whether to automatically tune hyperparameters"), default=False, action="store_true")
  parser.add_option('-i', '--iterations', help=default("Maximum iterations to run training"), default=3, type="int")
  parser.add_option('-r', '--randomizedselection', help=default("Use randomized data points"), default=False, action="store_true")

  options, otherjunk = parser.parse_args(argv)
  if len(otherjunk) != 0: raise Exception('Command line input not understood: ' + str(otherjunk))
  args = {}
  
  # Set up variables according to the command line input.
  print("Doing classification")
  print("--------------------")
  print("data:\t\t" + options.data)
  print("classifier:\t\t" + options.classifier)
  if not options.classifier == 'minicontest':
    print("using enhanced features?:\t" + str(options.features))
  else:
    print("using minicontest feature extractor")
  print("training set size:\t" + str(options.training))
  if(options.data=="digits"):
    printImage = ImagePrinter(DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorDigit
    else:
      featureFunction = basicFeatureExtractorDigit
    if (options.classifier == 'minicontest'):
      featureFunction = contestFeatureExtractorDigit
  elif(options.data=="faces"):
    printImage = ImagePrinter(FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT).printImage
    if (options.features):
      featureFunction = enhancedFeatureExtractorFace
    else:
      featureFunction = basicFeatureExtractorFace      
  else:
    print("Unknown dataset", options.data)
    print(USAGE_STRING)
    sys.exit(2)
    
  if(options.data=="digits"):
    legalLabels = list(range(10))
  else:
    legalLabels = list(range(2))
    
  if options.training <= 0:
    print("Training set size should be a positive integer (you provided: %d)" % options.training)
    print(USAGE_STRING)
    sys.exit(2)
    
  if options.smoothing <= 0:
    print("Please provide a positive number for smoothing (you provided: %f)" % options.smoothing)
    print(USAGE_STRING)
    sys.exit(2)
    
  if options.odds:
    if options.label1 not in legalLabels or options.label2 not in legalLabels:
      print("Didn't provide a legal labels for the odds ratio: (%d,%d)" % (options.label1, options.label2))
      print(USAGE_STRING)
      sys.exit(2)

  if(options.classifier == "naiveBayes" or options.classifier == "nb"):
    classifier = naiveBayes.NaiveBayesClassifier(legalLabels)
    classifier.setSmoothing(options.smoothing)
    if (options.autotune):
        print("using automatic tuning for naivebayes")
        classifier.automaticTuning = True
    else:
        print("using smoothing parameter k=%f for naivebayes" %  options.smoothing)
  elif(options.classifier == "perceptron"):
    classifier = perceptron.PerceptronClassifier(legalLabels,options.iterations)
  elif(options.classifier == "knnClassifier"):
    if(options.data == "faces"):
      classifier = knnClassifier_Faces.kNearestNeighborsClassifier()
    else:
      classifier = knnClassifier_Digits.kNearestNeighborsClassifier()
  elif(options.classifier == 'minicontest'):
    classifier = minicontest.contestClassifier(legalLabels)
  else:
    print("Unknown classifier:", options.classifier)
    print(USAGE_STRING)
    
    sys.exit(2)

  args['classifier'] = classifier
  args['featureFunction'] = featureFunction
  args['printImage'] = printImage


  
  return args, options

USAGE_STRING = """
  USAGE:      python dataClassifier.py <options>
  EXAMPLES:   (1) python dataClassifier.py
                  - trains the default mostFrequent classifier on the digit dataset
                  using the default 100 training examples and
                  then test the classifier on test data
              (2) python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
                  - would run the naive Bayes classifier on 1000 training examples
                  using the enhancedFeatureExtractorDigits function to get the features
                  on the faces dataset, would use the smoothing parameter equals to 2.5, would
                  test the classifier on the test data and performs an odd ratio analysis
                  with label1=3 vs. label2=6
                 """

# Main harness code

def runClassifier(args, options):

  featureFunction = args['featureFunction']
  classifier = args['classifier']
  printImage = args['printImage']
      
  # Load data 
  numTraining = options.training

  if(options.randomizedselection and options.data=='faces'):
     numTraining = 451
  if(options.randomizedselection and options.data=='digits'):
     numTraining = 5000
     

  if(options.data=="faces"):
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", numTraining,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numTraining)
    rawValidationData = samples.loadDataFile("facedata/facedatatrain", TEST_SET_SIZE,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", TEST_SET_SIZE)
    rawTestData = samples.loadDataFile("facedata/facedatatest", TEST_SET_SIZE,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", TEST_SET_SIZE)
  else:
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", numTraining,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    rawValidationData = samples.loadDataFile("digitdata/validationimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("digitdata/validationlabels", TEST_SET_SIZE)
    rawTestData = samples.loadDataFile("digitdata/testimages", TEST_SET_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", TEST_SET_SIZE)
    
  
  # Extract features
  print("Extracting features...")
  trainingData = list(map(featureFunction, rawTrainingData))
  validationData = list(map(featureFunction, rawValidationData))
  testData = list(map(featureFunction, rawTestData))
  
  if(options.randomizedselection):
    print("Using Randomized selection: selecting {} random data points from {}".format(options.training,numTraining))
    merged=list(map(lambda x, y:(x,y), trainingData, trainingLabels))
    random.shuffle(merged)
    trainingData,trainingLabels=[],[]
    for i in range(options.training):
      trainingData.append(merged[i][0])
      trainingLabels.append(merged[i][1])

  f = open("testdata_faces_simple.txt", "a")
  f.write("\n,".join(str(item) for item in testData))
  f.close()
  f = open("traindata_faces_simple.txt", "a")
  f.write('\n,'.join(str(item) for item in trainingData))
  f.close()
  # if classifier == KNN

  if(options.classifier == "knnClassifier"):
    accuracy, K = classifier.preprocessData(options.training)
    return accuracy, K
  else:
    # Conduct training and testing
    print("Training...")
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)

    print("Validating...")
    guesses = classifier.classify(validationData)
    correct1 = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print(str(correct1), ("correct out of " + str(len(validationLabels)) + ": %.1f%%") % (100.0 * correct1 / len(validationLabels)))

    print("Testing...")
    guesses = classifier.classify(testData)
    correct2 = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
    print(str(correct2), ("correct out of " + str(len(testLabels)) + ": %.1f%%") % (100.0 * correct2 / len(testLabels)))
    analysis(classifier, guesses, testLabels, testData, rawTestData, printImage)
    

  return correct1, correct2

if __name__ == '__main__':
  # Read inputt
  # Run classifier
  # t=0
  # for i in range(10):
  #   t+=500
  #   st=time.process_time()
  #   args, options = readCommand(['-d','digits','-c','perceptron','-t', str(t),'-k','1','-f'])
  #   # MAIN RUN : args, options = readCommand( sys.argv[1:] )
    
  #   correct1, correct2 = runClassifier(args, options)
  #   print("correcttt")
  #   print(correct1)
  #   print(correct2)
        
    
  #   et=time.process_time()
  #   print("Time: ",et-st)

  t=0
  validating_mean_acc = []
  validating_std_acc = []
  test_mean_acc = []
  test_std_acc = []

  # currently checking only for 10% data
  for i in range(2):
    t += 1000
    # MAIN RUN : args, options = readCommand( sys.argv[1:] )
    validating_acc = []
    test_acc = []
    bestK = []

    for j in range(2):
      st=time.process_time()
      args, options = readCommand(['-d','digits','-c','perceptron','-t', str(t),'-k','1','-f','-r'])
      #print(args)
      #print(options)
      correct1, correct2 = runClassifier(args, options)
      et=time.process_time()
      print("Time: ",et-st)
      validating_acc.append(correct1)
      test_acc.append(correct2)

    print(validating_acc)
    print(test_acc)
    print(bestK)
    isKnn = False
    if(test_acc[0] == 0):
      # knn
      test_mean_acc.append(np.mean(validating_acc))
      test_std_acc.append(np.std(validating_acc))
      isKnn = True
    else:
      validating_mean_acc.append(np.mean(validating_acc))
      validating_std_acc.append(np.std(validating_acc))
      test_mean_acc.append(np.mean(test_acc))
      test_std_acc.append(np.std(test_acc))

  if(isKnn):
    print("Testing : MEAN, STD")
    print(test_mean_acc)
    print(test_std_acc)
  else:
    print("Validation : MEAN, STD")
    print(validating_mean_acc)
    print(validating_std_acc)
    print("Testing : MEAN, STD")
    print(test_mean_acc)
    print(test_std_acc)
      
#python dataClassifier.py -c naiveBayes -d digits -t 1000 -f -o -1 3 -2 6 -k 2.5
# runClassifier(args, options)