### AUTHOR ###
# Name: Siddharth Agarwal
# Description:
# ---> This is the temp version of K-D Tree algorithm. Will be used in integration with the original LDC code
from tokenize import Double
import numpy as np
from sklearn.neighbors import KDTree
import time

def kd_Tree(X, D, N, n, point, Du, U):
    # ------------------------------ PART - A (Finding the n-nearest neighbours) ------------------------------

    #Reading N points from csv file and adding it in X (Numpy 2D Array).
    # N_points = open("pointcloud_coordinates.csv")
    # X = np.loadtxt(N_points, delimiter=",")

    # D = int(input("Enter the geometric dimensionility of the problem (D): "))
    if(D!=len(X[0])):
        print("ERROR: The Input pointcloud_coordinates.csv file contain points of dimension (D) = ", len(X[0]))
        print("Quit!!")
        quit()

    N = int(input("Enter the total number of points (N): "))
    if(N!=len(X)):
        print("ERROR: Total number of points in input pointcloud_coordinates.csv file (N) = ", len(X))
        print("Quit!!")
        quit()
    print("Reading the data from pointcloud_coordinates.csv file ...")

    # This is used if user has to enter all N points manually on terminal. Ignore, since input is given through csv file.
    # print("Enter all the point coordinates:")
    # X_mat = [list(map(float,input().split())) for i in range(N)]
    # X = np.array(X_mat)

    n = int(input("Enter the number of neighbour points (n): "))

    point = list(map(float, input("Enter the point of interest (x): ").split()))
    if(D!=len(point)): #Verifying whether the input point x has dimension D or not.
        print("ERROR: Point must be of dimension = ", D)
        print("Quit!!")
        quit()

    start = time.time()

    #The 2 lines below uses the python inbuilt library for KD-Tree.
    tree = KDTree(X)
    dist, ind = tree.query([point], k=n) 
    #dist is a 2-D vector which stores the distance value of all the 'n' neighbouring points from x.
    #ind is a 2-D vector which stores the indices of all the 'n' neighbouring points from x.

    end = time.time() # Timer is only used to check the time elapsed for finding the n-nearest neighbours from x.

    #Printing the result of Part A.
    print("\nNN Point Coordinates  ->  Distance from ", point)
    for i in range(0,n):
        print(X[ind[0][i]], "  ->  ", dist[0][i])

    print(f"Time taken by the program to find neighbours (in seconds): {end - start}\n")

    #Since all the weight values will be fixed throughout the code, therefore storing all the weights of n-nearest neighbours in array.
    weigth_arr = np.zeros(n, dtype = float)
    for i in range(0,n):
        if(dist[0][i]!=0):
            weigth_arr[i] = 1/(dist[0][i]**2)

    # ------------------------------ PART - A END ------------------------------




    # ------------------------------ PART - B (Finding the n-nearest neighbours) ------------------------------

    #Reading N points from csv file and adding it in X (Numpy 2D Array).
    # N_points = open("pointcloud_coordinates.csv")
    # X = np.loadtxt(N_points, delimiter=",")

    # D = int(input("Enter the geometric dimensionility of the problem (D): "))
    if(D!=len(X[0])):
        print("ERROR: The Input pointcloud_coordinates.csv file contain points of dimension (D) = ", len(X[0]))
        print("Quit!!")
        quit()

    N = int(input("Enter the total number of points (N): "))
    if(N!=len(X)):
        print("ERROR: Total number of points in input pointcloud_coordinates.csv file (N) = ", len(X))
        print("Quit!!")
        quit()
    print("Reading the data from pointcloud_coordinates.csv file ...")

    # This is used if user has to enter all N points manually on terminal. Ignore, since input is given through csv file.
    # print("Enter all the point coordinates:")
    # X_mat = [list(map(float,input().split())) for i in range(N)]
    # X = np.array(X_mat)

    n = int(input("Enter the number of neighbour points (n): "))

    point = list(map(float, input("Enter the point of interest (x): ").split()))
    if(D!=len(point)): #Verifying whether the input point x has dimension D or not.
        print("ERROR: Point must be of dimension = ", D)
        print("Quit!!")
        quit()

    start = time.time()

    #The 2 lines below uses the python inbuilt library for KD-Tree.
    tree = KDTree(X)
    dist, ind = tree.query([point], k=n) 
    #dist

    # ------------------------------ PART - B (Interpolation) ------------------------------

    #Reading all u points from csv file and adding it in U (Numpy 2D Array).
    # U_points = open("pointcloud_u_vector.csv")
    # U = np.loadtxt(U_points, delimiter=",")

    # Du = int(input("Enter the dimensionality of u(phi) function (Du): "))
    print("Reading the data from pointcloud_u_vector.csv file ...")

    #This array is used to store all the interpolated values, if Du >= 2.
    interpolated_ux_arr = np.zeros(Du, dtype = float)

    #Case-1: if u is a scalar.
    if(Du==1):
        interpolated_ux = 0
        ux_numer = 0
        ux_denom = 0
        flag_val = 0 #To check whether the interpolation function follows the condition when distance == 0.
        for i in range(0,n):
            if(dist[0][i]==0):
                interpolated_ux = U[ind[0][i]]
                flag_val = 1
                break
            else:
                ux_numer = ux_numer + (weigth_arr[i]*U[ind[0][i]])
                ux_denom = ux_denom + weigth_arr[i]
        
        if(flag_val==1):
            print("Interpolated u(x) = ", interpolated_ux)
        elif(flag_val==0 and ux_denom!=0):
            print("Interpolated u(x) = ", ux_numer/ux_denom)

    #Case-2: if u is a vector and has more than one components.
    else:
        for j in range(0,Du):
            ux_numer = 0
            ux_denom = 0
            flag_val = 0
            for i in range(0,n):
                if(dist[0][i]==0):
                    interpolated_ux_arr[j] = U[ind[0][i]][j]
                    flag_val = 1
                    break
                else:
                    ux_numer = ux_numer + (weigth_arr[i]*U[ind[0][i]][j])
                    ux_denom = ux_denom + weigth_arr[i]
                    
            if(flag_val==0 and ux_denom!=0):
                interpolated_ux_arr[j] = ux_numer/ux_denom

            print("Interpolated u(x) component", j+1, " = ", interpolated_ux_arr[j])

    print()

    # ------------------------------ PART - B END ------------------------------
