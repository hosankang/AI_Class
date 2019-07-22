import numpy as np
import matplotlib.pyplot as plt

def activation_func(n):
    return ((n >= 0) + -1 * (n < 0)).astype(np.int)

w1 = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [1, -1], [-1, 1], [1, -1], [-1, 1], [-1, -1], [1, 1], [1, 1]], int)
b1 = np.array([[-2], [3], [0.5], [0.5], [-1.75], [2.25], [-3.25], [3.75], [6.25], [-5.75], [-4.75]], float)
w2 = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
               [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1]], int)
b2 = np.array([[-3], [-3], [-3], [-3]], int)
w3 = np.array([1, 1, 1, 1], int)
b3 = np.array([3], int)

'''
#3개의 레이어 설정
x = input()
y = input()
X = np.array([[x], [y]], float)
a1 = activation_func(np.dot(w1, X) + b1)
a2 = activation_func(np.dot(w2, a1) + b2)
a3 = activation_func(np.dot(w3, a2) + b3)
'''
###레이어1 그래프
xx=np.arange(0,10)
yy=[]
i=0
while i<11:
    yy.append(-w1[i][1]*(w1[i][0]*xx+b1[i]))
    i=i+1
i=0
while i<11:
    plt.plot(xx,yy[i])
    i = i+1
plt.axis([0,6,0,4])
plt.show()

# 랜덤한 50개의 점
import random

for i in range(0, 50):
    x = random.random() * 5
    y = random.random() * 3

    X = np.array([[x], [y]], float)
    a1 = activation_func(np.dot(w1, X) + b1)
    a2 = activation_func(np.dot(w2, a1) + b2)
    a3 = activation_func(np.dot(w3, a2) + b3)
    if a3 == 1:
        plt.plot(x, y, 'o', mec='k', mfc='k', ms=12)
    else:
        plt.plot(x, y, 'o', mec='k', mfc='w', ms=12)

x1 = []
y1 = []
p = [[0, 0, 1, 1, 4, 4, 5, 5, 6, 6, 7, 7, 4, 4, 7, 7], [2, 3, 3, 2, 10, 8, 8, 10, 8, 10, 10, 8, 9, 8, 8, 9]]
for j in range(0, 16):
    m1 = -w1[p[0][j]][1] * w1[p[0][j]][0]
    m2 = -w1[p[1][j]][1] * w1[p[1][j]][0]
    d1 = -w1[p[0][j]][1] * b1[p[0][j]]
    d2 = -w1[p[1][j]][1] * b1[p[1][j]]
    x1.append((d2 - d1) / (m1 - m2))
    y1.append(m1 * ((d2 - d1) / (m1 - m2)) + d1)

X1 = []
Y1 = []
for i in range(0, 4):
    X1.append(x1[i])
    Y1.append(y1[i])
    plt.fill(X1, Y1, facecolor='dodgerblue')
X1.clear()
Y1.clear()
for i in range(4, 8):
    X1.append(x1[i])
    Y1.append(y1[i])
    plt.fill(X1, Y1, facecolor='dodgerblue')
X1.clear()
Y1.clear()
for i in range(8, 12):
    X1.append(x1[i])
    Y1.append(y1[i])
    plt.fill(X1, Y1, facecolor='dodgerblue')
X1.clear()
Y1.clear()
for i in range(12, 16):
    X1.append(x1[i])
    Y1.append(y1[i])
    plt.fill(X1, Y1, facecolor='dodgerblue')

plt.show()