"""
Instructions:

This assignment is to "simulate" Google's pagerank as discussed in class.
This is a skeleton program where you need to add your code in places.

The program first reads web.txt to load the page network. Each row refers to a page
and 0/1 indicates which other pages it visits to (1 mean visit, else no).

It then populates transition probability matrix, P. Recall that a page visits to 
the pages it has links to with uniform probability, and with some residual probability 
it visits to any other page (may be to itself) uniformly at random. The parameter Alpha defines this split.  

Given P, the program then analytically finds ranks of pages (i.e., pi's of underlying Markov chain
of pages). It also "simulates" a navigation process to compute the same. 

The program then computes the difference between the two measurements and show them in a plot.

Add your codes at designated places. 

Answer the following question at the end of your program

Q. Change the seed (currently 100) to different values. Do you see changes in results? 
Can you explain why? Can you measure how much variation you see?

WORK INDEPEDENTLY. CODE SHARING IS STRICTLY PROHIBITED, AND IF HAPPENS WILL LEAD TO PENALTY.
"""

import numpy as np
import random
import matplotlib.pyplot as plt

def printP(P):
    for row in range(len(P)):
        for col in range(len(P[row])):
            print P[row][col], 
        print     


# Compute transition probability matrix
def populateP(file_name):
    alpha = 0.85
    P = None
    
    with open(file_name) as readSource:
        page_id=0
        total_pages = int(readSource.readline())
        P = [[0 for x in range(total_pages)] for y in range(total_pages)]
        for x in range(total_pages):
            num=(readSource.readline())
            cnt=num.count('1')
            for y in range(total_pages):
                if int(num[2*y])==1:
                    P[x][y]=alpha/cnt
                
                P[x][y]+=(1-alpha)/total_pages
    
    #printP(P)         
    return P
            

def computeDiff(pi, simpi):
    if len(pi) != len(simpi):
        raise Error('Pi dimension does not match!!')
    
    sumdiff = 0    
    for i in range(len(pi)):
        sumdiff += abs(pi[i] - simpi[i])
    return sumdiff
    

# Compute rank analytically                
def solveForPi(P): 
    # TODO: solve steady state equations using np.linalg
    # Hint: formuate the problem in a matrix form as A*pi = B, then solve for pi using numpy's linalg
    # Your task is to construct A and B as appropriate.
    
    A = None
    B = None
    
    # Your code here
    A = [[0 for x in range(len(P))] for y in range(len(P[0]))]
    for x in range(len(P)):
        for y in range(len(P)):
            if(x!=y):
                A[y][x]=-P[x][y]
            else:
                A[y][x]=1.0-P[x][y]
    
    B=[0.0 for x in range(len(P))]
    for x in range(len(P)):
        A[len(P)-1][x]=1.0;
    B[len(P)-1]=1.0;
    pi = np.linalg.solve(A, B)
    
    
    print pi
    #F=open("output.txt",'w')
    #F.write(str(pi))
    return pi


# Compute rank by simulation
# Visit pages for 'iterations' number of times

mark=None

def probabilityCalc(indx,iterations,P):
    global mark
    while iterations!=0:
        mark[indx]+=1
        indx=choosePage(P[indx])
        iterations-=1

def computeRankBySimulation(P, iterations):        
    
    simPi = [0 for i in range(len(P))]
   
    # TODO: start navigation through pages
    # You can start from page 0
    # You can use the function choosePage below to choose a page 
    # at random with a given probability distribution 
    
    # Your code here
    
    global mark
    mark=[0 for i in range(len(P))]
    
    num=random.randint(0,len(P)-1)
    probabilityCalc(num, iterations,P)
    for x in range(len(mark)):
        simPi[x]=1.0*mark[x]/iterations
    
    return simPi
     
        
"""
    Sample X as defined by distribution "prob"
    prob is a list of probability values, such as [0.2, 0.5, 0.3]. 
    These are the values index positions take, that is, O happens with prob 0.2, 1 with 0.5 and so on.
"""
def choosePage(prob): 
    U = random.random()
    P = 0
    for i in range(len(prob)):
        P = P + prob[i]
        if U < P:
            return i     
    return len(prob)
        
        
# main function
def main():
    P = populateP('web.txt')
    
    # Compute rank analytically
    analyticalPi = solveForPi(P)
    
    # Compute rank by "simulation"
    
    random.seed(10000)
    
    X = []
    Y = []
    for itr in range(1, 11):    
        itr = itr * 10000
        simulatedPi = computeRankBySimulation(P, itr)
        print "%d\t%f" %(itr, computeDiff(analyticalPi, simulatedPi))
        X.append(itr / 1000)
        Y.append(computeDiff(analyticalPi, simulatedPi))
    
    plt.plot(X, Y)
    plt.xlabel("Iterations (1000's)")    
    plt.ylabel("Pi difference")
    plt.show()


if __name__ == "__main__":
    main()


"""
Your answer for the question goes here.
...
We can see change is result. Different PRNG functions show different number sequence for a seed value.
Python uses Mersenne twister as pseudorandom number generator. Same seed in this function always
generates same number sequence though no particular cycle is not found in the sequence. Change is seed
thus conveys changes in random number sequence which creates different outcome, but always same for same seed.
Higher seed valu with higher number of iteration seems to provide better answer with this graph but may not
do so for different link graph.

"""
