"""
Name: gonn.py (gene ontology neural network)
Author: Oliver Giles
Date: March 2017
Description:
	- Takes training and control FASTA files as input
	- Formats the first 2/3 of those FASTA files into a combined training dataset
	- Formats the final 1/3 of those FASTA files into a combined testing dataset
	- Trains a neural network using the training dataset
	- Tests the neural network using the testing dataset, returning a success rate
	- Displays a graph of the predictive accuracy of the network as it is trained
"""


from Bio import SeqIO
import tensorflow as tf
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt


"""
ARGUMENT HANDLING
"""


parser = argparse.ArgumentParser(description=
	"""Provide valid FASTA files. The training file should contain sequences with
	the feature you wish the neural network to detect. The control should contain 
	sequences that do not have this feature. The more diversified the control data,
	the better.""")

#Define command line arguments.
parser.add_argument("-t", 
	nargs = 1,
	help = "specify a training FASTA file",
	required = True,
	type = argparse.FileType('r'))			#Setting the type allows argparse to handle IOError exceptions.
parser.add_argument("-c", 
	nargs = 1,
	help = "specify a control FASTA file",
	required = True,
	type = argparse.FileType('r'))

#Parse arguments provided.
args = parser.parse_args()


"""
VARIABLES
"""


l1nodeCount = 400 #The number of neurons in layer 1
l2nodeCount = 400 
l3nodeCount = 400

outputCount = 2 #Our seqs either have the desired gene ontology or do not

batchSize = 100

epochCount = 15


"""
FUNCTION DEFINITIONS
"""


def neural_network_graph(data):
	#DEFINE THE STRUCTURE OF THE NEURAL NET
	#Create dictionaries with tensors of initially random weights and zeroed biases for each layer of connections
	global l1Dict, l2Dict, l3Dict, lOutDict
	l1Dict = {'weights': tf.Variable(tf.random_normal([tensorSize, l1nodeCount])),
	'biases':tf.Variable(tf.zeros(l1nodeCount))}
	l2Dict = {'weights': tf.Variable(tf.random_normal([l1nodeCount, l2nodeCount])),
	'biases':tf.Variable(tf.zeros(l2nodeCount))}
	l3Dict = {'weights': tf.Variable(tf.random_normal([l2nodeCount, l3nodeCount])),
	'biases':tf.Variable(tf.zeros(l3nodeCount))}
	lOutDict = {'weights': tf.Variable(tf.random_normal([l3nodeCount, outputCount])),
	'biases':tf.Variable(tf.zeros(outputCount))}

	"""
	Below, we multiply the inputs of our layers by their weights, then add the biases (Wx + b).
	We then apply our activation function, relu, to the results. Each neuron is 'activated' if the result of
	this function is greater than zero. It then feeds forward a value akin to the frequency of an actual neuron
	which is used as the input of the next layer using the same Wx + b formula. So, for example, l1, after 
	having the relu function applied, is a tensor of values between 0 and 1. Feeding these values forward
	into another layer creates a so-called 'deep' neural network. 
	"""
	l1 = tf.add(tf.matmul(data, l1Dict['weights']), l1Dict['biases'])
	l1 = tf.nn.relu(l1)
	l2 = tf.add(tf.matmul(l1, l2Dict['weights']), l2Dict['biases'])
	l2 = tf.nn.relu(l2)
	l3 = tf.add(tf.matmul(l2, l3Dict['weights']), l3Dict['biases'])
	l3 = tf.nn.relu(l3)

	#The output layer results in two values: likelihoods of the input tensor having/not having the GO feature
	lOut = tf.add(tf.matmul(l3, lOutDict['weights']), lOutDict['biases'])
	return lOut

def train_neural_network(data):
	#DEFINE THE TRAINING PROCESS AND RUN EPOCHS
	prediction = neural_network_graph(data)

	#Compare the outcome of running data through our network to our classifications
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y) +
		0.01*tf.nn.l2_loss(l1Dict['weights']) +
    	0.01*tf.nn.l2_loss(l1Dict['biases']) +
    	0.01*tf.nn.l2_loss(l2Dict['weights']) +		#l2 regularisation helps prevent overfitting to the training data
    	0.01*tf.nn.l2_loss(l2Dict['biases']) +		#which should allow for more epochs of training
    	0.01*tf.nn.l2_loss(l3Dict['weights']) +
    	0.01*tf.nn.l2_loss(l3Dict['biases']) +
    	0.01*tf.nn.l2_loss(lOutDict['weights']) +
    	0.01*tf.nn.l2_loss(lOutDict['biases'])) 

	"""
	Below, we do the actual training of the network. This involves using an optimisation method
	to try and reduce the difference between our expected outcome and our actual outcome (loss). 
	The optimiser modifies the variable tensors (weights and biases) to move towards
	lower loss using a step value that is itself optimised to change over time.
	"""
	
	optimise = tf.train.AdamOptimizer().minimize(loss)

	#Compare outputs against predicted outcome to determine accuracy
	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

	trainingPlotData = [] #Arrays for storing accuracy over time to plot on graph
	testingPlotData = []

	#Begin TensorFlow session
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		#For each epoch (complete passing over of the data)
		for epoch in range(epochCount):
			epochLoss = 0 			

			#Split the data into batches to be fed to the network. Larger batches use more memory but 
			#also make the optimisation less erratic. 
			i = 0
			while i < len(training_x):
				batchStartIndex = i
				batchEndIndex = i + batchSize

				batch_x = np.array(training_x[batchStartIndex:batchEndIndex])
				batch_y = np.array(training_y[batchStartIndex:batchEndIndex])

				#Feed our batch into the network
				optimiseResult, lossResult = sess.run([optimise, loss], feed_dict={x: batch_x, y: batch_y})
				epochLoss += lossResult

				trainingPlotData.append(accuracy.eval({x: batch_x, y: batch_y}))
				testingPlotData.append(accuracy.eval({x: testing_x, y: testing_y}))

				i += batchSize

			print('Epoch ' + str((epoch + 1)) + '. Loss: ' + str(epochLoss) + '.')

		#Run our testing data through the network and print our final accuracy!
		print 'Accuracy:', accuracy.eval({x: testing_x, y: testing_y})

		return trainingPlotData, testingPlotData

def format_FASTA(training, control):
	tSeqCount = 0
	cSeqCount = 0
	
	#Count the records
	for record in SeqIO.parse(training, "fasta"):
		tSeqCount += 1
	training.seek(0)	#Reset file position so we can iterate over it again

	for record in SeqIO.parse(control, "fasta"):
		cSeqCount += 1
	control.seek(0)

	#Use the smaller number so we can have an equal amount of control/training data
	if tSeqCount <= cSeqCount:
		#2/3 of the data will be used to train, and 1/3 to test
		trainCount = 2 * (tSeqCount / 3)
		testCount = tSeqCount - trainCount
	else:
		trainCount = 2 * (cSeqCount / 3)
		testCount = cSeqCount - trainCount

	#Create alphabet of all unique codons and ordered codon pairings
	codons = ['A', 'R', 'N', 'D', 'C',
			'Q', 'E', 'G', 'H', 'I',
			'L', 'K', 'M', 'F', 'P',
			'S', 'T', 'W', 'Y', 'V',
			'U', 'O']

	alphabet = list(codons) #Just using '=' acts like a pointer, so use list function

	for codon in codons:
		for secondCodon in codons:
			alphabet.append(codon + secondCodon)

	#Each sequence will be represented using a tensor with a frequency for each letter in the alphabet
	global tensorSize 
	tensorSize = len(alphabet)

	dataset_training = []
	dataset_testing = []

	#For each sequence, count frequency of each element in alphabet
	for fastaFile in [training, control]:
		recordCount = 0
		for record in SeqIO.parse(fastaFile, 'fasta'):
			if recordCount <= (trainCount + testCount):
				#Create an array of tensorSize to store the alphabet frequencies
				freqs = np.zeros(tensorSize)
				classification = np.zeros(2)

				for idx, char in enumerate(record.seq):
					#Find the corresponding index in alphabet, and add 1 to that index in freqs
					alphabetIndex = alphabet.index(char)
					freqs[alphabetIndex] += 1

					#Repeat for ordered pairings, checking that idx + 1 is not out of range
					try:
						alphabetIndex = alphabet.index(char + record.seq[idx+1])
						freqs[alphabetIndex] += 1
					except IndexError:
						pass

				#Check whether the sequence is training or control and adjust classification tensor
				if fastaFile == training:
					classification = [0, 1]
				else: 
					classification = [1, 0]

				if recordCount < trainCount:
					#Output data to training dataset
					freqs_ = list(freqs)
					dataset_training.append([freqs, classification])
				else:
					#Output data to testing dataset
					freqs_ = list(freqs)
					dataset_testing.append([freqs, classification])

				recordCount += 1

	#Shuffle our data to prevent skewed learning
	random.shuffle(dataset_training)
	random.shuffle(dataset_testing)

	dataset_training = np.array(dataset_training)
	dataset_testing = np.array(dataset_testing)

	#Split our frequencies (x) and classifications (y) so they can be plotted against each other
	training_x = list(dataset_training[:,0])
	training_y = list(dataset_training[:,1])
	testing_x = list(dataset_testing[:,0])
	testing_y = list(dataset_testing[:,1])

	#Return formatted data for feeding into the neural network
	return training_x, training_y, testing_x, testing_y

def plot_accuracy(x1, x2):
	plt.plot(x1, label = 'training accuracy')
	plt.plot(x2, label = 'testing accuracy', color = 'r')
	plt.xlabel('Batches')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.grid()
	plt.show()


"""
EXECUTION
"""


#Format the provided FASTA files for use in the neural network.
training_x, training_y, testing_x, testing_y = format_FASTA(args.t[0], args.c[0])

#Define the tensor placeholders that will store our data as it is fed through
x = tf.placeholder('float', [None, tensorSize]) #Our input vector is flat - essentially a list - and tensorSize in length
y = tf.placeholder('float')

#Train the neural network...
x1, x2 = train_neural_network(x)

#Plot a graph of the training results
plot_accuracy(x1, x2)