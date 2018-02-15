#!/usr/bin/env python

import numpy as np
from numpy.linalg import inv
import cv2
import random
import os 
import matplotlib.pyplot as plt
import scipy as sp
from scipy.spatial.distance import mahalanobis
from scipy.spatial.distance import cdist

if __name__ == "__main__":
	#########################################PCA###############################
	#Step 1: PCA:	
	people_total = 40
	faces_dir = 'att_faces/'
	train_count = 6
	test_faces_count = 4

	t = train_count * people_total
	m = 92
	n = 112
	mn = m * n
	p = t
	#total= mn*p
	
	#Training set
	training_ids = []  	
	T = np.empty(shape=(mn, t), dtype='float64')
	cur_img = 0
	#print "training"
	for face_id in xrange(1, people_total + 1):
		training_id_curr = random.sample(range(1, 11), train_count)
		training_ids.append(training_id_curr)
		for training_id in training_id_curr:
			path_to_img = os.path.join(faces_dir, 's' + str(face_id), str(training_id) + '.pgm')
			img = cv2.imread(path_to_img, 0)  #as reading grayscale img
			img_col = np.array(img, dtype='float64').flatten()  #making it mn*1 col matrix
			T[:, cur_img] = img_col[:] #storing it in a column in mn*t database
			cur_img += 1
	mean_img_col = np.sum(T, axis=1) / t 

	for j in xrange(0, t):
		T[:, j] -= mean_img_col[:]
	cov = np.matrix(T.transpose()) * np.matrix(T)
	cov /= t

	evalues, evectors = np.linalg.eig(cov)
	sort_indices = evalues.argsort()[::-1]
	evalues = evalues[sort_indices]
	evectors = evectors[sort_indices]	
	k = 100         #i.e. "k" as in "k" highest eigen values
	
	x_list = []	
	y_list = []	
	
	for m_best in range(10,100,15):
	
		featureVector = (evectors[0:k]).transpose() #p*k
		eigenFaces = featureVector.transpose() * T.transpose()
		signature = eigenFaces * T #k*p
		#print signature.shape()
		#print "training done"
	
		#STEP 2: 
		no_of_classes = t/train_count #(6*40)/6 = 40 classes
		#print "no of classes=", no_of_classes

		#STEP 3:
		mean_of_classes = np.empty(shape=(k, no_of_classes), dtype='float64')
	
		for i in range(no_of_classes):
			mean_curr = np.sum(signature[:,i*train_count:(i+1)*train_count], axis=1) / train_count
			#print mean_curr.shape()		
			mean_of_classes[:,i] = np.reshape(mean_curr[:,:],k)
	
		#print mean_of_classes.shape
	
		M = np.sum(mean_of_classes,axis=1)/no_of_classes
		#print M.shape
	
		#STEP 4:within class scatter matrix
		C = no_of_classes
		SW = np.empty(shape=(k, k), dtype='float64')
		SB = np.empty(shape=(k, k), dtype='float64')
		for i in range(C):
			V = np.array(signature[:,i*train_count:(i+1)*train_count])
			for j in range(train_count):
				V[:,j] = V[:,j] - mean_of_classes[:,i]
			SW = SW + np.dot((V),(np.transpose(V)))
			#print SW.shape
			SB = SB + np.dot(mean_of_classes[:,i] - M,np.transpose(mean_of_classes[:,i]))
			#print SB.shape

		#STEP 5: FIND CRITERION FUNCTION
		J = np.dot(inv(SW),SB)
		#print J.shape

		#STEP 6: Find Eigen vector and Eigen values of the Criterion function
		eigVal, eigVect = np.linalg.eig(J)

		#STEP 7,8: find m best principal components
		sort_indices = eigVal.argsort()[::-1]
		eigVal = eigVal[sort_indices]
		eigVect = eigVect[sort_indices]

		#print eigVect.shape #k*k
	
		W = (eigVect[0:m_best]).transpose()
		#print W.shape #k*m
	
		#STEP 9: Generate the fisher faces (FF)
		FF = np.dot(np.transpose(W), signature) #m*p i.e. 10*(6*40)
		#print FF.shape

		##################TESTING ###########	

		test_count = test_faces_count * people_total
		test_correct = 0

		for face_id in xrange(1, people_total + 1):
			for test_id in xrange(1, 11):
				if (test_id in training_ids[face_id-1]) == False:  #selecting left over 4 faces
					path_to_img = os.path.join(faces_dir, 's' + str(face_id), str(test_id) + '.pgm')
					img = cv2.imread(path_to_img, 0)
					img_col = np.array(img, dtype='float64').flatten() #make it 1*mn

					#step 2: Do mean Zero, by subtracting mean face (M) to this test face					
					img_col -= mean_img_col
					img_col = np.reshape(img_col, (mn, 1))      #make it column vector mn*1
			
					#step 3:Project this mean aligned face to eigenfaces

					PEF = eigenFaces * img_col #PEF
					ProjFischerTestImg = np.dot(np.transpose(W),PEF) #m*1
					#print projected_Fisher_Test_Img.shape

					#Method 1: Euclidean distance
					#norms = np.linalg.norm(FF - ProjFischerTestImg,axis=0) 
					#Method 2: Mahalanobis distance
					min = np.inf		
					#print FF.shape			
					for temp in range(FF.shape[1]):
						#print "temp=",temp
						#print FF[:,temp].shape
						XA = np.reshape(FF[:,temp],m_best)
						XB = np.reshape(ProjFischerTestImg,m_best)
						#print XA.shape
						#print XB.shape
						results = cdist(XA, XB,'mahalanobis')
						#print results
						norms = np.diag(results)					
						if(norms < min):
							min = norms

					closest_face_id = min
					print "MIN=",min
					answer = (closest_face_id / train_count) + 1 
				
					if answer == face_id:
							test_correct += 1
							#print "image:",path_to_img," result: correct"
					#else:
						#print "image:",path_to_img," result: wrong, got ",answer
		#print "TESTING COMPLETE!"
		accuracy = float(100. * test_correct / test_count)
		print 'm_best:',m_best,'Correct: ' + str(accuracy) + '%'
		y_list.append(accuracy)
		x_list.append(m_best)

plt.plot(x_list, y_list, 'g-')
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,100))
plt.show()

