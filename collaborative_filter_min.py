'''
A python file that calls the slaves from the master.
'''

#We first import the neccessary scientific libraries such as numPy, sciPy and matplotlib.
import numpy as np
import scipy.optimize as opt
import csv
import scipy
import sys
from normalizeRatings import normalizeRatings
from mpi4py import MPI

#Load the list of movies and split by movie id
def loadMovieList():
    movieList = []

    ## Read the fixed movieulary list
    with open('movie_ids.txt') as fid:
        for line in fid:
            movieName = line.split(' ', 1)[0].strip()
            movieList.append(movieName)

    return movieList

#Load the list of movies as a string
def loadMovieNameList():
    movieList = []

    ## Read the fixed movieulary list
    with open('movie_ids.txt') as fid:
        for line in fid:
            movieName = line
            movieList.append(movieName)

    return movieList


#Load the matrix of movies and users
reader = csv.reader(open("in1.csv", "rU"), delimiter=',')
a = list(reader)
y = np.array(a).astype('float')
y=y.T
size_y = np.shape(y)
print np.shape(y)


num_movies = size_y[0]
num_users = size_y[1]

#Set number of features.
num_features = 10

#Create a matrix R of ones and zeros
R = np.array(y!=0).astype('float')
size_r = np.shape(R)




movieList = loadMovieList()

size_y= np.shape(y)
num_movies=size_y[0]
num_users = size_y[1]

#set overflow parameters
lda=1






#set number of nodes
nprocs=4	

X = np.asarray(np.random.randn(num_movies, num_features));
Theta = np.asarray(np.random.randn(num_users, num_features));

TTheta_grad = (np.zeros((num_users, num_features)))
XX_grad = (np.zeros((num_movies-(num_movies%nprocs), num_features)))

#Check if number of movies is directly divisible by number of processors
if num_movies%nprocs > 0:
	T_xsize=num_movies-int(num_movies/nprocs)*nprocs;
	T = np.zeros((num_movies%nprocs, num_features))
	#print np.shape(T)


#Spawn the four slaves.
# The slaves will execute the file grad_2.py
#From this point on the code will execute parallely along with the file grad_2.py
comm = MPI.COMM_SELF.Spawn(sys.executable, args=['grad_2.py'], maxprocs=nprocs)
N = np.array([num_movies, num_users, num_features, lda])



comm.Bcast([N, MPI.INT], root=MPI.ROOT)
 

#we split the matrix as (num_movies/num_nodes)
#if num_movies is not divisible by num_nodes then we let the last node add the movies that were the remainder or extras.
#example 1602 movies by 4 nodes is node divisible(remainder is 2 movies). the fourth node will also take the remaining 2 movies into account.
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
	
	#Send the variables that will remain constant accross 100 iterations.		
	comm.Send([Tx, MPI.FLOAT], dest=i, tag=0)
	comm.Send([TY, MPI.FLOAT], dest=i, tag=0)
	comm.Send([TR, MPI.INT], dest=i, tag=0)
	

#Cost variable to be reduced
Jprime=0#np.array([0.0])

#Overflow parameter to be reduced
reg_Xprime=0#np.array([0.0])


for iteration in range(0,100):

 
    

    TTheta_grad = (np.zeros((num_users, num_features)))
    XX_grad = (np.zeros((num_movies-(num_movies%nprocs), num_features)))
    Jprime = np.array([0.0])
    reg_Xprime=np.array([0.0])

    #broadcast user probability matrix to all users

    comm.Bcast([np.ascontiguousarray(Theta), MPI.FLOAT], root=MPI.ROOT)
    

    #reduce the cost from the slaves and sum them up.
    comm.Reduce(None, [Jprime, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
    #reduce the overflow from the slaves and sum them up.
    comm.Reduce(None, [reg_Xprime, MPI.DOUBLE], op=MPI.SUM, root=MPI.ROOT)
    
    #reg_Xprime = np.array([sum((Tx**2).flatten(1))])
  
    #calculate reg_theta
    reg_theta = np.asarray(Theta)**2
   
    #Sum up the variables as per the formula of the costfunction
    Jprime = Jprime/(2*num_users*num_movies) + lda/2*(np.sum(reg_theta.flatten(1)) + reg_Xprime)
   
  
    reg_theta = np.sum((Theta**2).flatten(1))
    reg_theta = (Theta**2)

    print Jprime
   

  
    #Reduce the gradient for the movie probabililty by gathering individual slave values.  
    comm.Gather(None, [XX_grad, MPI.DOUBLE], root=MPI.ROOT)

    #If movies were divisble by process, gather the extra movies computed by the last node
    if num_movies%nprocs > 0:	

	comm.Recv([T, MPI.DOUBLE], source=nprocs-1,tag=1)
	XX_grad = np.concatenate((XX_grad,T))
    X=XX_grad
    #Calculate late the new user probability matrix using the movie probility matrix
    Theta = np.ascontiguousarray(np.linalg.solve(np.dot(X.T, X) + lda * np.eye(num_features),  np.dot(X.T, y)).T)

    
    #print TTheta_grad
    
     		
    #Wait for master and slave to synchronize
    comm.Barrier()
comm.Disconnect()
#MPI job complete

#Calculate the predicated matrix 
p = np.dot(X, Theta.T)


size_predicted = np.shape(p)
print '#####################################################'

print size_predicted

#Load movie names
movieList = loadMovieNameList()
#Set printo options
np.set_printoptions(threshold='nan')
#Write Top ten predicted movies to a file for each suer
for i in range(0, size_predicted[1]):
	f1 = open('results/user_'+str(i)+'.txt', 'w+')
	my_predictions = (p[:,i]).flatten(1)
	ix = np.argsort(my_predictions)
	#print np.shape(p[:, i])
	
	#print '***********************************************'
	#print ix[:-11:-1]
	for j in ix[:-11:-1]:
		firstspace = movieList[j].find(' ')
		f1.write('{0}: {1}' .format(movieList[j][0:firstspace], movieList[j][firstspace+1: len(movieList[j])]))
		




