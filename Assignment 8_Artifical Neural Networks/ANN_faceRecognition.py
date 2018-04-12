#!/usr/bin/env python

import numpy as np
import cv2
import random
import os 
import matplotlib.pyplot as plt
from math import exp

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

def activate(weights, inputs):
	activation = weights[-1] #last weight is the bias
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

#sigmoid function
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))
 
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def transfer_derivative(output):
	return output * (1.0 - output)

def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))


if __name__ == "__main__":
	
	#Step 1: Create Face database:	
	people_total = 40
	faces_dir = 'att_faces/'
	train_count = 6
	test_faces_count = 4

	t = train_count * people_total
	m = 92
	n = 112
	mn = m * n
	p = people_total*10
	#total= mn*p
	
	#Training set
	training_ids = []  	
	T = np.empty(shape=(mn, t), dtype='float64')
	cur_img = 0
	print "TRAINING SET:"
	for face_id in xrange(1, people_total + 1):
		#print face_id
		training_id_curr = random.sample(range(1, 11), train_count) #pick 6 items randomly for each person
		training_ids.append(training_id_curr)
		#print "Face id=",face_id,"Image Id=",training_id_curr	
		for training_id in training_id_curr:
			path_to_img = os.path.join(faces_dir, 's' + str(face_id), str(training_id) + '.pgm')
			img = cv2.imread(path_to_img, 0)  #as reading grayscale img
			#print path_to_img
			img_col = np.array(img, dtype='float64').flatten()  #making it mn*1 col matrix
			T[:, cur_img] = img_col[:] #storing it in a column in mn*t database
			cur_img += 1
	#print T
	#Step 2: Mean Calculation
	mean_img_col = np.sum(T, axis=1) / t 

	#Step 3: Subtract mean face from all training faces. Now they are mean aligned faces
	for j in xrange(0, t):
		T[:, j] -= mean_img_col[:]
	#print T
	#Step 4: Co-Variance of the Mean aligned faces(Turk and peterland: p*p instead of mn*mn)
	C = np.matrix(T.transpose()) * np.matrix(T)
	C /= t
	
	#print C

	#step 5: eigenvalue and eigenvector decomposition:
	evalues, evectors = np.linalg.eig(C)
	#print evalues
	#step 6: Find the best direction (Generation of feature vectors)
	
	#sort acc to descending eigen value
	sort_indices = evalues.argsort()[::-1]
	evalues = evalues[sort_indices]
	evectors = evectors[sort_indices]	
	#print evalues
	evalues_count = 100         #i.e. "k" as in "k" highest eigen values
	
	x_list = []	
	y_list = []	
	featureVector = (evectors[0:evalues_count]).transpose() #p*k
	
	#Step 7: Generating Eigenfaces
	eigenFaces = featureVector.transpose() * T.transpose()
	#Step 8: Generate Signature of Each Face:
	signature = eigenFaces * T #k*p

 	K = evalues_count
	p = 240

	signature = signature.tolist()

	print len(signature)
	
	list_temp = []
	class_ = 0
	flag = 0
	for i in range(len(signature)):
		flag = flag
			list_temp.append(class_)
		class_ = class_+1
	signature.append(list_temp)	
	print len(signature)
	print signature
	
	dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]

	

	n_inputs = len(dataset[0]) - 1
	n_outputs = len(set([row[-1] for row in dataset]))
	network = initialize_network(n_inputs, 2, n_outputs)
	train_network(network, dataset, 0.5, 20, n_outputs)

	#dataset = signature
	#n_inputs = len(dataset[0])
	#n_outputs = len(set([row[-1] for row in dataset]))

	#Step 1: Make test database
	results_file = os.path.join('answer', 'att_results.txt') 	
	f = open(results_file, 'w') 

	test_count = test_faces_count * people_total
	test_correct = 0

	for face_id in xrange(1, people_total + 1):
		for test_id in xrange(1, 11):
			if (test_id in training_ids[face_id-1]) == False:  #selecting left over 4 faces
				path_to_img = os.path.join(faces_dir, 's' + str(face_id), str(test_id) + '.pgm')
				img = cv2.imread(path_to_img, 0)
				img_col = np.array(img, dtype='float64').flatten() #make it 1*mn
				dataset = img_col
				n_inputs = len(dataset[0]) - 1
				n_outputs = len(set([row[-1] for row in dataset]))
				
				

	accuracy = float(100. * test_correct / test_count)
	print 'Correct: ' + str(accuracy) + '%'
	f.write('Correct: %.2f\n' % (accuracy))
	f.close()
	y_list.append(accuracy)
	x_list.append(evalues_count)

		

	plt.plot(x_list, y_list, 'g-')
	x1,x2,y1,y2 = plt.axis()
	plt.axis((x1,x2,0,100))
	plt.show()	







