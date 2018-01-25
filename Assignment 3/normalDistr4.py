import numpy as np
import random
import matplotlib.pyplot as plt
import math

P1 = 0.3
P2 = 0.4
P3 = 0.3

#u1 = np.matrix('0;0')
#u2 = np.matrix('0;1')
#E1 = np.matrix('1 0;0 1')

u1 = [0, 0]
u2 = [0, 1]
u3 = [1, 0]
E1 = [[1,0],[0,1]]
E2 = [[2,-1],[-1,1.5]]
E3 = [[2,1],[1,1.5]]

n = [100, 500, 1000, 2000, 5000]
#n = [100]
error_calc = np.zeros(len(n))
error_act  = np.zeros(len(n))
index = 0

for num in n:
	artificial=[]
	cls1 = []
	cls2 = []
	cls3 = []

	for i in range(num):
		randNo = random.uniform(0,1)
		
		if(randNo <= P1):
			x1, y1 = np.random.multivariate_normal(u1, E1, 1).T
			artificial.append([x1[0],y1[0],'1'])	
			cls1.append([x1[0],y1[0]])
		elif(randNo>P1 and randNo<(P1+P2)):
			x1, y1 = np.random.multivariate_normal(u2, E2, 1).T
			artificial.append([x1[0],y1[0],'2'])	
			cls2.append([x1[0],y1[0]])
		else:
			x1, y1 = np.random.multivariate_normal(u3, E3, 1).T
			artificial.append([x1[0],y1[0],'3'])
			cls3.append([x1[0],y1[0]])
	
	print len(artificial)
	print len(cls1)
	print len(cls2)
	print len(cls3)
	
	artificial = np.array(artificial)
	cls1 = np.array(cls1)
	cls2 = np.array(cls2)
	cls3 = np.array(cls3)

	#plt.plot(cls1[:,0], cls1[:,1], 'ro', cls2[:,0], cls2[:,1], 'bo')	
	#plt.show()	
	
	n2 = int(num/2)
	train = artificial[0:n2]
	test = artificial[n2:num]
	
	print len(train)
	print len(test)
	
	cls1_train, cls2_train, cls3_train = 0,0,0	
	cov1 = np.zeros((2,2))
	cov2 = np.zeros((2,2))	
	cov3 = np.zeros((2,2))	
	mean = np.zeros((3,2)) #1st row is for class 1-x,y and 2nd for class 2

	for i in train:
		if (i[2] == "1"):
			mean[0][0] = mean[0][0] + float(i[0])
			mean[0][1] = mean[0][1] + float(i[1])
			cls1_train = cls1_train + 1			
			
		elif(i[2] == "2"):
			mean[1][0] = mean[1][0] + float(i[0])
			mean[1][1] = mean[1][1] + float(i[1])
			cls2_train = cls2_train + 1
		else:
			mean[2][0] = mean[2][0] + float(i[0])
			mean[2][1] = mean[2][1] + float(i[1])
			cls3_train = cls3_train + 1
	
	mean[0] = mean[0]/cls1_train
	mean[1] = mean[1]/cls2_train
	mean[2] = mean[2]/cls3_train
	
	print "ESTIMATED VALUES:"
	print "Mean of class 1:", mean[0]
	print "Mean of class 2:", mean[1]
	print "Mean of class 3:", mean[2]

	for i in train:
		for j in range (0,2):
			for k in range (0,2):
				if (i[2] == "1"):
					cov1[j][k] = cov1[j][k] + (float(i[j])-mean[0][j])*(float(i[k])-mean[0][k])
				elif(i[2] == "2"):	
					cov2[j][k] = cov2[j][k] + (float(i[j])-mean[1][j])*(float(i[k])-mean[1][k])
				else:
					cov3[j][k] = cov3[j][k] + (float(i[j])-mean[2][j])*(float(i[k])-mean[2][k])		
	
	cov1 = cov1/cls1_train
	cov2 = cov2/cls2_train
	cov3 = cov3/cls3_train
	
	print
	print "Covariance of class 1:"
	print cov1
	print "Covariance of class 2:"
	print cov2
	print "Covariance of class 3:"
	print cov3

	P1_est = float(cls1_train)/float(len(train))
	P2_est = float(cls2_train)/float(len(train))
	P3_est = float(cls3_train)/float(len(train))

	print 
	print "Prior Probability of class 1=",P1_est
	print "Prior Probability of class 2=",P2_est
	print "Prior Probability of class 3=",P3_est

	miscls = 0 #no of misclassifications
	
	mean = np.array(mean)
	cov1 = np.array(cov1)
	cov2 = np.array(cov2)
	cov3 = np.array(cov3)

	calc_cls = 0

	for i in test:
		test_data = [float(i[0]), float(i[1])]
		test_data = np.array(test_data)
		c1 = np.matmul(np.matmul((test_data - mean[0]), np.linalg.inv(cov1)),(test_data - mean[0]).T);
		c2 = np.matmul(np.matmul((test_data - mean[1]), np.linalg.inv(cov2)),(test_data - mean[1]).T);
		c3 = np.matmul(np.matmul((test_data - mean[2]), np.linalg.inv(cov3)),(test_data - mean[2]).T);		
		p1 = 1/math.sqrt(np.linalg.det(cov1)) * math.exp(-0.5*c1) 
		p2 = 1/math.sqrt(np.linalg.det(cov2)) * math.exp(-0.5*c2)
		p3 = 1/math.sqrt(np.linalg.det(cov3)) * math.exp(-0.5*c3)
		if(P1_est*p1 >= P2_est*p2 and P1_est*p1>=P3_est*p3):
			calc_cls = 1
		elif(P2_est*p2 >= P1_est*p1 and P2_est*p2>=P3_est*p3):
			calc_cls = 2
		else:
			calc_cls = 3
		actual_cls = int(i[2])
		if(calc_cls != actual_cls):
			miscls = miscls+1 
	print "Misclassifications(est)=", float(miscls)/float(len(test))

	miscls2 = 0

	for i in test:
		test_data = [float(i[0]), float(i[1])]
		test_data = np.array(test_data)
		c1 = np.matmul(np.matmul((test_data - u1), np.linalg.inv(E1)),(test_data - u1).T);
		c2 = np.matmul(np.matmul((test_data - u2), np.linalg.inv(E2)),(test_data - u2).T);
		c3 = np.matmul(np.matmul((test_data - u3), np.linalg.inv(E3)),(test_data - u3).T);
		p1 = 1/math.sqrt(np.linalg.det(cov1)) * math.exp(-0.5*c1) 
		p2 = 1/math.sqrt(np.linalg.det(cov2)) * math.exp(-0.5*c2)
		p3 = 1/math.sqrt(np.linalg.det(cov3)) * math.exp(-0.5*c3)
		if(P1_est*p1 >= P2_est*p2 and P1_est*p1>=P3_est*p3):
			calc_cls = 1
		elif(P2_est*p2 >= P1_est*p1 and P2_est*p2>=P3_est*p3):
			calc_cls = 2
		else:
			calc_cls = 3
		actual_cls = int(i[2])
		if(calc_cls != actual_cls):
			miscls2 = miscls2+1 
	print "Misclassifications(act)=", float(miscls2)/float(len(test))
	
	error_calc[index] = float(miscls)/float(len(test))
	error_act[index] = float(miscls2)/float(len(test))
	index = index+1

plt.plot(n, error_calc, 'r-', n, error_act, 'g-')	
x1,x2,y1,y2 = plt.axis()
plt.axis((x1,x2,0,1))
plt.show()	
