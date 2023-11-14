import numpy as np
from scipy.optimize import linprog

c = [-1, 2, -3, 1]  
A = [[2, -1, 2, -3], [1, 2, -1, 1]]  
b = [5, 3]  

x1=(0, 4)
x2=(0, 4)
x3=(0, 4)
x4=(0, 4)

res = linprog(c, A_ub=A, b_ub=b, bounds=(x1, x2, x3, x4), method='simplex', options={"disp": True})
print(res)