import numpy as np
import matplotlib.pyplot as plt
import math
def ActivationFunction1(n):
    return 1/(1+math.e**(-n))
def ActivationFunction2(n):
    return n
w11=10
w12=10
b1=-10
b2=10
w21=1
w22=1
b3=0
p=np.arange(-2,2,0.01)
p1=ActivationFunction1(w11*p+b1)
p2=ActivationFunction1(w12*p+b2)
p3=ActivationFunction2((p1*w21)+(p2*w22)+b3)
plt.ylabel("a^2")
plt.xlabel("p")
#plt.axis([-2,2,-2,2])
print(p3)
plt.plot(p,p3,'k')
plt.show()
