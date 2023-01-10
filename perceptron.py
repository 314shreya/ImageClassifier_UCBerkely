import util
class PerceptronClassifier:
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter() 

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.features = trainingData[0].keys()
        for iteration in range(self.max_iterations):
            print ("Starting iteration ", iteration, "...")
            for data_index in range(len(trainingData)): 
              val_y = self.classify([trainingData[data_index]])[0]
              training_lab_curr=trainingLabels[data_index]
              if val_y != trainingLabels[data_index]:
                curr_weight=self.weights
                curr_weight[training_lab_curr] = curr_weight[training_lab_curr] + trainingData[data_index]  
                curr_weight[val_y] = curr_weight[val_y] - trainingData[data_index]

    def classify(self, data):
        guesses = []
        for dat in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * dat
            guesses.append(vectors.argMax())
        return guesses