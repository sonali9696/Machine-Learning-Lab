#!/usr/bin/env python

import os
import cv2
import sys
import shutil
import random
import numpy as np


if __name__ == "__main__":
	
	#Face database:	
	people_total = 40
	faces_dir = '.'
	train_count = 6
	test_faces_count = 4

	l = train_count * people_total
	m = 92
	n = 112
	mn = m * n
	p = people_total*10
	#total= mn*p
	
	#Training set
	training_ids = []  	
	L = np.empty(shape=(mn, l), dtype='float64')
	cur_img = 0
	for face_id in xrange(1, people_total + 1):

            training_ids = random.sample(range(1, 11), train_count) #pick 6 items randomly for each person
            training_ids.append(training_ids) 

            for training_id in training_ids:
                path_to_img = os.path.join(faces_dir, 's' + str(face_id), str(training_id) + '.pgm')          
                print '> reading file: ' + path_to_img

                img = cv2.imread(path_to_img, 0)  #as reading grayscale img                           
                img_col = np.array(img, dtype='float64').flatten()  #making it mn*1 col matrix

                L[:, cur_img] = img_col[:] 
                cur_img += 1

