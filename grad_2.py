#!/usr/bin/env python

#import the scientific libraries
from mpi4py import MPI
import numpy as np

#Point of entry for the slaves
comm = MPI.Comm.Get_parent()
size = comm.Get_size() #number of nodes
rank = comm.Get_rank() #since the code is executed by each slave, the rank represents the slave node number

N = np.array([0,0,0,0])


comm.Bcast([N, MPI.INT], root=0)
#Get the size of the matrices and the number of features and the overflow parameters
num_movies=N[0]
num_users=N[1]
num_features=N[2]
lda = N[3]

#user probability matrix does not change
Theta = (np.zeros((num_users, num_features)))

#split the matrices as mention in collaborative_filter_min.py
#extras or remainder of movies are handled by the last processor
if rank !=size-1:
	Tx = (np.zeros((int(num_movies/size), num_features)))
	TY = (np.zeros((int(num_movies/size), num_users)))
	TR = (np.zeros((int(num_movies/size), num_users)))
	size_tx = int(num_movies/size)
else:
	Tx = (np.zeros((int(num_movies/size+num_movies%size), num_features)))
	TY = (np.zeros((int(num_movies/size+num_movies%size), num_users)))
	TR = (np.zeros((int(num_movies/size+num_movies%size), num_users)))
	size_tx = int(num_movies/size+num_movies%size)

#receive the data sent fromt the master
comm.Recv(Tx, source=0, tag=0)
comm.Recv(TY, source=0, tag=0)
comm.Recv(TR, source=0, tag=0)


for iteration in range(0,100):
	#print Tx
    	
	comm.Bcast([Theta, MPI.FLOAT], root=0)
	
	#Cost is (X*Theta_transpose - Y)^2. But X and Y are split based on number of nodes
	mat = np.dot(Tx, Theta.T)-TY

	mat_sq = mat**2
	#Multiply by R matrix to taken into account only movies 'actually' rated by users
	temp = mat_sq*TR

	#Calculate the temporary cost and temporary overflow value
	Jprime = np.array([sum((0.1*temp).flatten(1))])
	reg_Xprime = np.array([sum((Tx**2).flatten(1))])
	
	
	#Reduce the two above mention values back to the master
	comm.Reduce([np.ascontiguousarray(Jprime), MPI.DOUBLE], None, op=MPI.SUM, root=0)
	comm.Reduce([np.ascontiguousarray(reg_Xprime), MPI.DOUBLE], None,op=MPI.SUM, root=0)
	
	#CALCULATING GRADIENT

	
	temp = mat*TR
	#calculate the new X value by applying gradient descent. The X(movie probability matrix) value is partitioned accross nodes.	
	XX_grad = np.ascontiguousarray(np.linalg.solve(np.dot(Theta.T, Theta) + lda * np.eye(num_features), np.dot(TY, Theta).T).T)
  	
	#TTheta_grad = np.ascontiguousarray(np.linalg.solve(np.dot(Tx.T, Tx) + lda * np.eye(num_features),  np.dot(Tx.T, TY)).T)

	#Reduce the new X value to the master
	comm.Gather([XX_grad, MPI.DOUBLE], None, root=0) 

	
	#print TTheta_grad[0][0]
	
	#Send the extras calculated by the last node to the master.
	if num_movies%size >0 and rank==size-1:
		extra_x,extra_y = np.shape(XX_grad)
	
		T = XX_grad[extra_x - num_movies%size:extra_x, :]
	
		comm.Send([T, MPI.DOUBLE], dest=0, tag=1)
		#Tx = np.concatenate((Tx,T))
	Tx=XX_grad
	#wait for synchronization
	comm.Barrier()	
comm.Disconnect()
