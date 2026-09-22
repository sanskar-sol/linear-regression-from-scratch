import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

#visualling the data 
# plt.scatter(data.age, data.charges)
# plt.show()

#loss function
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].age
        y = points.iloc[j].charges
        total_error += (y - (m * x + b)) ** 2
    
    avg_error = total_error / float(len(points))
    
    return avg_error

#gradient descent
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0  
    
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].age
        y = points.iloc[i].charges
        
        m_gradient = x * (y - (m_now * x + b_now))
        b_gradient = y - (m_now * x + b_now)
        
        m_gradient = (m_gradient * -2) / float(n)
        b_gradient = (b_gradient * -2) / float(n)
        
    m_now = m_now - m_gradient * L
    b_now = b_now - b_gradient * L 
    
    return m_now, b_now

# actual regression

m = 0
b = 0
L = 0.001 # learning rate
iterations = 500 # also called epoch

for i in range(iterations):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, L)
    
print(f"Final m: {m}, Final b: {b}")

plt.scatter(data.age, data.charges, color="blue")
plt.plot( list(range(15, 90)), [m * x + b for x in range(15, 70)])
plt.show()