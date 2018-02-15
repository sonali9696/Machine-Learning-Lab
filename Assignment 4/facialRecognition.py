#!/usr/bin/env python

import numpy as np
import cv2
import random
import os 
import matplotlib.pyplot as plt

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
	#print "TRAINING SET:"
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
	#evalues_count = 100         #i.e. "k" as in "k" highest eigen values
	
	x_list = []	
	y_list = []	
	for evalues_count in range(1,240,22):	
	
		featureVector = (evectors[0:evalues_count]).transpose() #p*k
		
		#Step 7: Generating Eigenfaces
		eigenFaces = featureVector.transpose() * T.transpose()
		#Step 8: Generate Signature of Each Face:
		signature = eigenFaces * T #k*p


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

					#step 2: Do mean Zero, by subtracting mean face (M) to this test face					
					img_col -= mean_img_col

					img_col = np.reshape(img_col, (mn, 1))      #make it column vector mn*1
				
					#step 3:Project this mean aligned face to eigenfaces

					projected_test_face = eigenFaces * img_col 		
			
					norms = np.linalg.norm(signature - projected_test_face,axis=0) #calculates euclidean
					closest_face_id = np.argmin(norms)
					answer = (closest_face_id / train_count) + 1 

					if answer == face_id:
						test_correct += 1
						f.write('image: %s\nresult: correct\n\n' % path_to_img)
						#print "image:",path_to_img," result: correct"
					else:
						f.write('image: %s\nresult: wrong, got %2d\n\n' %(path_to_img, answer))
						#print "image:",path_to_img," result: wrong, got ",answer

		#print "TESTING COMPLETE!"
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

	######################IMPOSTER#
	face_id = 41
	for test_id in xrange(1, 11):
		if(test_id <=5):
			path_to_img = os.path.join(faces_dir, 's' + str(face_id), str(test_id) + '.jpg')
		else:
				path_to_img = os.path.join(faces_dir, 's' + str(face_id), str(test_id) + '.pgm')
		
		img = cv2.imread(path_to_img, 0)
		img_col = np.array(img, dtype='float64').flatten() 
		img_col -= mean_img_col
		img_col = np.reshape(img_col, (mn, 1))     
		projected_test_face = eigenFaces * img_col 		

		norms = np.linalg.norm(signature - projected_test_face,axis=0) #calculates euclidean
		closest_face_id = np.argmin(norms)
		answer = (closest_face_id / train_count) + 1 

		if answer == face_id:
			test_correct += 1
			f.write('image: %s\nresult: correct\n\n' % path_to_img)
			#print "image:",path_to_img," result: correct"
		else:
			f.write('image: %s\nresult: wrong, got %2d\n\n' %(path_to_img, answer))
			#print "image:",path_to_img," result: wrong, got ",answer

	
		










