#!/usr/bin/python
import sys
import numpy as np
import math
import random as rnd
import time

def rmi(string):
    p = string.find(':')
    l = len(string)
    return float(string[p+1:l])

def parseVector(line):
    return [rmi(x) for x in line]

def kernel(a, b, n, gamma):
    sum = 0
    for i in range(0, n):
        sum = sum + (a[i]-b[i])**2
    #sum = np.sum((a - b) ** 2)
    return math.exp(-1.0*gamma*sum)


def selfkernel():
    return 1.0

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print >> sys.stderr, "Usage: python linsmo.py <input_file> n m"

##fjr read data
    fh = open(sys.argv[1],"r")
    m = int(sys.argv[2])
    n = int(sys.argv[3])
    y = []
    data = [[0 for i in range(n)] for j in range(m)]

    lines = fh.readlines()
    #len(lines[0].split(' '))-1
    i = 0
    for l in lines:
        words = l.strip().split(' ')
        flag = int(words[0])
        if(flag < 1):
            flag = -1
        y.append(flag)
        for word in words[1:]:
            splitflag = word.find(':')
            l = len(word)
            value = float(word[splitflag+1:l])
            key = int(word[0:splitflag])-1
            #print "[DEBUF]", "i is ",i, "label is ", label ,"value is ",value
            data[i][key] = value
        i = i + 1
    #print(data)
    #print y

    print "size is m, n: ", m, n
    cost = 1.0
    gamma = 1/float(n)

    tolerance = 1e-3
    eps = 1e-3
    tau = 1e-12
    cEpsilon = cost - eps
    #print 'number of samples is: ' + str(m) +  ' number of features is: ' + str(n)
    #print label
    #print data
    #print "FJR test scale: ", kernel(data[1], data[1], n,  gamma )
    
    start = time.clock()

    A = []
    G = []

    for i in range(0, m):
        G.append(-1.0)
        A.append(0)
   

    iteration = 0
    while(1):
        #(i,j) = selectB()
        i = -1
        G_max = -float("Inf")
        G_min = float("Inf")

        for t in range(m):
            if ( (y[t] == 1 and A[t] < cost) or (y[t] == -1 and A[t] > 0) ):
                if (-y[t]*G[t] >= G_max):
                    i = t
                    G_max = -y[t]*G[t]
        j = -1
        obj_min = float("Inf")
        for t in range(m):
            if ((y[t] == 1 and A[t] > 0) or (y[t] == -1 and A[t] < cost)):
                b = G_max + y[t]*G[t]
                if (-y[t]*G[t] <= G_min):
                    G_min = -y[t]*G[t]
                if (b > 0):
                    a = kernel(data[i],data[i],n,gamma)+kernel(data[t],data[t],n,gamma)-2*y[i]*y[t]*y[i]*y[t]*kernel(data[i],data[t],n,gamma)
                    if (a <= 0):
                        a = tau
                    if (-(b*b)/a <= obj_min):
                        j = t
                        obj_min = -(b*b)/a
                    
        if (G_max-G_min < eps):
            (i,j) = (-1,-1)
        #select end
        if(j == -1):
            break

        a = kernel(data[i],data[i],n,gamma)+kernel(data[j],data[j],n,gamma)-2*kernel(data[i],data[j],n,gamma)
        
        if (a <= 0):
            a = tau

        b = -y[i]*G[i]+y[j]*G[j]

        oldAi = A[i]
        oldAj = A[j]
        A[i] = A[i] + y[i]*b/a
        A[j] = A[j] - y[j]*b/a

        sum = y[i]*oldAi+y[j]*oldAj
        if A[i] > cost:
            A[i] = cost
        if A[i] < 0:
            A[i] = 0

        A[j] = y[j]*(sum-y[i]*A[i])

        if A[j] > cost:
            A[j] = cost
        if A[j] < 0:
            A[j] = 0
        A[i] = y[i]*(sum-y[j]*A[j])

        deltaAi = A[i] - oldAi 
        deltaAj = A[j] - oldAj
        for t in range(m):
            G[t] = G[t] + y[t]*y[i]*kernel(data[t],data[i],n,gamma)*deltaAi+y[t]*y[j]*kernel(data[t],data[j],n,gamma)*deltaAj

        #print "iteration ",iteration
        iteration = iteration + 1

    end = time.clock()
    print "runtime is ",end-start
    print "iteration is ", iteration

    outfile = open("tmplibsvm.out",'w')

    nSV = 0
    pSV = 0
    for i in range(0, m):
        if (A[i] > 0):
            if (y[i] > 0):
                pSV = pSV + 1;
            else:
                nSV = nSV + 1;
    printGamma = False;
    printCoef0 = False;
    printDegree = False;
    degree = None
    kernelType = "rbf";
    if (kernelType == "polynomial"):
        printGamma = True;
        printCoef0 = True;
        printDegree = True;
    elif (kernelType == "rbf"):
        printGamma = True;
    elif (kernelType == "sigmoid"):
        printGamma = True;
        printCoef0 = True;
    
    outfile.write("svm_type c_svc\n");
    outfile.write("kernel_type " + str(kernelType) + "\n");
    if (printDegree):
        outfile.write("degree " + str(degree) + "\n");
    if (printGamma): 
        outfile.write("gamma " + str(gamma) + "\n");
    if (printCoef0):
        outfile.write("coef0 " + str(coef0) + "\n");

    outfile.write("nr_class 2\n");
    outfile.write("total_sv " + str(nSV + pSV) + "\n");
    outfile.write("rho " + "XXX" + "\n");
    outfile.write("label 1 -1\n");
    outfile.write("nr_sv " + str(pSV) + " " + str(nSV) + "\n");
    outfile.write("SV\n");

    for i in range(0, m):
        if (A[i] > 0):
            outfile.write(str(y[i]*A[i])+" ");
            for j in range(0, n): 
                outfile.write(str(j+1) + ":"+str(data[i][j]) + " ");
            outfile.write("\n");
    outfile.close();   