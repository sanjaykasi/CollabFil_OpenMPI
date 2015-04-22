import numpy as np
import scipy.optimize as opt
import csv
import scipy
from mpi4py import MPI
import mpi4py.rc 
import sys

def costFunction(theta_in, *args):
	y=args[0]
	R=args[1]
	num_movies=args[2]
	num_users=args[3]
	num_features=args[4]
	lda=args[5]
	J=0
   
   
 	X = (theta_in[ 0:num_movies*num_features].reshape((num_features, num_movies)).T)
	Theta = (theta_in[ num_movies*num_features: num_movies*num_features+num_users*num_features].reshape((num_features, num_users)).T)

	Jprime=np.array([0.0])
	reg_Xprime=np.array([0.0])
	nprocs=1

	comm = MPI.COMM_SELF.Spawn(sys.executable, args=['cost.py'], maxprocs=nprocs)

	
	print 'start cost'
	N = np.array([num_movies, num_users, num_features, lda])
	comm.Bcast([N, MPI.INT], root=MPI.ROOT)
	comm.Bcast([np.ascontiguousarray(Theta), MPI.FLOAT], root=MPI.ROOT)


	for i in range(0,nprocs):
		if i != nprocs-1:
			Tx = (np.ascontiguousarray(X[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs), :]))
			TY = (np.ascontiguousarray(y[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs), :]))
			TR = (np.ascontiguousarray(R[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs), :]))
		else:
			extras = num_movies%nprocs
			Tx = (np.ascontiguousarray(X[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs)+extras, :]))
			TY = (np.ascontiguousarray(y[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs)+extras, :]))
			TR = (np.ascontiguousarray(R[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs)+extras, :]))
				
		comm.Send([Tx, MPI.FLOAT], dest=i, tag=0)
		comm.Send([TY, MPI.FLOAT], dest=i, tag=0)
		comm.Send([TR, MPI.INT], dest=i, tag=0)
  	
	comm.Reduce(None, [Jprime, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
	comm.Reduce(None, [reg_Xprime, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
	reg_Xprime = np.array([sum((Tx**2).flatten(1))])

	reg_theta = np.asarray(Theta)**2

    	Jprime = Jprime/2 + lda/2*(np.sum(reg_theta.flatten(1)) + reg_Xprime)
   
	print 'end cost'

	return Jprime[0]

def grad(theta_in, *args):
	y=args[0]
	R=args[1]
	num_movies=args[2]
	num_users=args[3]
	num_features=args[4]
	lda=args[5]

	#CHANGE THIS
   	X = (theta_in[0:num_movies*num_features].reshape((num_features, num_movies)).T)
	Theta = (theta_in[num_movies*num_features: num_movies*num_features+num_users*num_features].reshape((num_features, num_users)).T)
        

	nprocs=1

	comm = MPI.COMM_SELF.Spawn(sys.executable, args=['child.py'], maxprocs=nprocs)
		
	
	N = np.array([num_movies, num_users, num_features, lda])
	

	TTheta_grad = (np.zeros((num_users, num_features)))
	XX_grad = (np.zeros((num_movies-(num_movies%nprocs), num_features)))
	if num_movies%nprocs > 0:
		T_xsize=num_movies-int(num_movies/nprocs)*nprocs;
		T = np.zeros((T_xsize, num_features))
	print 'start grad'
	comm.Bcast([N, MPI.INT], root=MPI.ROOT)
	comm.Bcast([np.ascontiguousarray(Theta), MPI.FLOAT], root=MPI.ROOT)

	

	#print Theta
	for i in range(0,nprocs):
		if i != nprocs-1:
			Tx = (np.ascontiguousarray(X[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs), :]))
			TY = (np.ascontiguousarray(y[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs), :]))
			TR = (np.ascontiguousarray(R[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs), :]))
		else:
			extras = num_movies%nprocs
			Tx = (np.ascontiguousarray(X[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs)+extras, :]))
			TY = (np.ascontiguousarray(y[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs)+extras, :]))
			TR = (np.ascontiguousarray(R[i*int(num_movies/nprocs):(i+1)*int(num_movies/nprocs)+extras, :]))
				
		comm.Send([Tx, MPI.FLOAT], dest=i, tag=0)
		comm.Send([TY, MPI.FLOAT], dest=i, tag=0)
		comm.Send([TR, MPI.INT], dest=i, tag=0)
			
	
		
	comm.Reduce(None, [TTheta_grad, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
	comm.Gather(None, [XX_grad, MPI.DOUBLE], root=MPI.ROOT)
	if num_movies%nprocs > 0:	
		comm.Recv([T, MPI.DOUBLE], source=nprocs-1,tag=1)
		XX_grad = np.concatenate((XX_grad,T))
	
	print 'end grad'
	

	comm.Disconnect()
		
	theta_grad = np.squeeze(np.asarray(np.concatenate((XX_grad.flatten(1), TTheta_grad.flatten(1)))))
    
    
	return theta_grad


'''
#Load X
reader = csv.reader(open("movies.csv", "rb"), delimiter=',')
a = list(reader)
X = np.mat(a).astype('float')
size_x = np.shape(X)
num_movies = size_x[0]
print np.shape(X)


#Load Theta
reader = csv.reader(open("users.csv", "rU"), delimiter=',')
a = list(reader)
Theta = np.mat(a).astype('float')
size_theta = np.shape(Theta)
num_users = size_theta[0]
print np.shape(Theta)

num_features = size_x[1]
'''
#Load R
reader = csv.reader(open("R.csv", "rU"), delimiter=',')
a = list(reader)
R = np.mat(a).astype('float')
size_r = np.shape(R)
print np.shape(R)

#Load Y
reader = csv.reader(open("y.csv", "rU"), delimiter=',')
a = list(reader)
y = np.mat(a).astype('float')
size_y = np.shape(y)
print np.shape(y)

num_movies = size_y[0]
num_users = size_y[1]

num_features = 10

#theta_in = np.concatenate((np.array(X.flatten(1), Theta.flatten(1)))
#print np.shape(theta_in)





#X_grad = theta_grad[0:num_movies*num_features].reshape((num_features, num_movies)).T
#theta_grad = theta_grad[ num_movies*num_features: num_movies*num_features+num_users*num_features].reshape((num_features, num_users)).T



X = np.asarray(np.random.randn(num_movies, num_features));
Theta = np.asarray(np.random.randn(num_users, num_features));

#print X_grad

initial_parameters =  np.concatenate((X.flatten(1), Theta.flatten(1)))



#x = lambda t:costFunction(t, y, R, num_movies, num_users, num_features, 10)
args = (y, R, num_movies, num_users, num_features, 10)
theta_grad = scipy.optimize.fmin_cg(costFunction, initial_parameters,fprime=grad, args = args, maxiter=2, disp=True)

X_grad = theta_grad[0:num_movies*num_features].reshape((num_features, num_movies)).T
theta_grad = theta_grad[num_movies*num_features: num_movies*num_features+num_users*num_features].reshape((num_features, num_users)).T


#print np.shape(X_grad)
#print X_grad










