import numpy as np

# 1) Create a vector of size 10 with all elements equal to zero
v = np.zeros(10, dtype=np.int)
print(v)
# 2) Create a 3 by 3 matrix with numbers from 0 to 8. Multiply all elements in the matrix by 3
w = np.arange(0,9,1).reshape((3,3))
print(w)
w_3 = 3*w
print(w_3)
# 3) Calculate the matrix product A transpose times B
A = np.array([[1,4],[2,5],[3,6]])
B=np.array([[1,0],[0,1],[0,0]])
R1=np.dot(A.T, B)
print(R1)
# 4) Add the row vector [-1 1] to each row of A_T*B
R2 = np.array([-1,1])+R1
print(R2)
''' 5) Create a 3-by-3 matrix X with random values in the range [0; 1). Create a binary matrix
    C of size 3-by-3 where the value of an element is False if the value of the corresponding
    element in X is less than 0.5, and True otherwise. Set all values in X at indices where C
    is True to -1.
'''
R3 = np.random.rand(3,3)
print(R3)
R4 = R3 > 0.5
print(R4)
R3[R3 > 0.5] = -1
print(R3)