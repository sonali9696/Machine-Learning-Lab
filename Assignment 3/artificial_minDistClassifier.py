import numpy as np
import random
import matplotlib.pyplot as plt
import math

#min dist can be used only if sigma is equal to identity in both classes

P1 = 0.5
P2 = 0.5

u1 = [0, 0]
u2 = [0, 1]
E1 = [[1,0],[0,1]]


n = [100, 500, 1000, 2000, 5000]
#n = [100]
error_calc = np.zeros(len(n))
error_act  = np.zeros(len(n))
index = 0

for num in n:
	artificial=[]
	cls1 = []
	cls2 = []

	for i in range(num):
		randNo = random.uniform(0,1)
		
		if(randNo <= P1):
			x1, y1 = np.random.multivariate_normal(u1, E1, 1).T
			artificial.append([x1[0],y1[0],'1'])	
			cls1.append([x1[0],y1[0]])
				
		else:
			x1, y1 = np.random.multivariate_normal(u2, E1, 1).T
			artificial.append([x1[0],y1[0],'2'])
			cls2.append([x1[0],y1[0]])
	
	print len(artificial)
	print len(cls1)
	print len(cls2)
	
	artificial = np.array(artificial)
	cls1 = np.array(cls1)
	cls2 = np.array(cls2)

	#plt.plot(cls1[:,0], cls1[:,1], 'ro', cls2[:,0], cls2[:,1], 'bo')	
	#plt.show()	
	
	n2 = int(num/2)
	train = artificial[0:n2]
	test = artificial[n2:num]
	
	print len(train)
	print len(test)
	
	cls1_train, cls2_train = 0,0	
	cov1 = np.zeros((2,2))
	cov2 = np.zeros((2,2))		
	mean = np.zeros((2,2)) #1st row is for class 1-x,y and 2nd for class 2

	for i in train:
		if (i[2] == "1"):
			mean[0][0] = mean[0][0] + float(i[0])
			mean[0][1] = mean[0][1] + float(i[1])
			cls1_train = cls1_train + 1			
			
		else:
			mean[1][0] = mean[1][0] + float(i[0])
			mean[1][1] = mean[1][1] + float(i[1])
			cls2_train = cls2_train + 1
	
	mean[0] = mean[0]/cls1_train
	mean[1] = mean[1]/cls2_train
	
	print "ESTIMATED VALUES:"
	print "Mean of class 1:", mean[0]
	print "Mean of class 2:", mean[1]

	
	for i in train:
		for j in range (0,2):
			for k in range (0,2):
				if (i[2] == "1"):
					cov1[j][k] = cov1[j][k] + (float(i[j])-mean[0][j])*(float(i[k])-mean[0][k])
				else:	
					cov2[j][k] = cov2[j][k] + (float(i[j])-mean[1][j])*(float(i[k])-mean[1][k])
		
	cov1 = cov1/cls1_train
	cov2 = cov2/cls2_train
	
	print
	print "Covariance of class 1:"
	print cov1
	print "Covariance of class 2:"
	print cov2

	P1_est = float(cls1_train)/float(len(train))
	P2_est = float(cls2_train)/float(len(train))

	print 
	print "Prior Probability of class 1=",P1_est
	print "Prior Probability of class 2=",P2_est

	miscls = 0 #no of misclassifications
	
	mean = np.array(mean)
	cov1 = np.array(cov1)
	cov2 = np.array(cov2)

	for i in test:
		test_data = [float(i[0]), float(i[1])]
		test_data = np.array(test_data)
	
		p1 = np.linalg.norm(test_data-mean[0])
		p2 = np.linalg.norm(test_data-mean[1])
		if(p1 <= p2):
			calc_cls = 1
		else:
			calc_cls = 2
		actual_cls = int(i[2])
		if(calc_cls != actual_cls):
			miscls = miscls+1 
	print "Misclassifications(est)=", float(miscls)/float(len(test))

	miscls2 = 0

	for i in test:
		test_data = [float(i[0]), float(i[1])]
		test_data = np.array(test_data)
		p1 = np.linalg.norm(test_data-u1)
		p2 = np.linalg.norm(test_data-u2)
		if(p1 <= p2):
			calc_cls = 1
		else:
			calc_cls = 2
		actual_cls = int(i[2])
		if(calc_cls != actual_cls):
			miscls2 = miscls2+1 
	print "Misclassifications(calc)=", float(miscls2)/float(len(test))
	
	error_calc[index] = float(miscls)/float(len(test))
	error_act[index] = float(miscls2)/float(len(test))
	index = index+1

plt.plot(n, error_calc, 'r-', n, error_act, 'g-')	
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,1))
plt.show()	
