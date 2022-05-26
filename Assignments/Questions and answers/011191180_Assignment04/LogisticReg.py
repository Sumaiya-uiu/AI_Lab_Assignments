

from google.colab import drive
drive.mount("/content/gdrive")

from numpy import genfromtxt
data_path = '/content/gdrive/MyDrive/DataSet/Iris/iris.csv'
iris_data = genfromtxt(data_path, delimiter=',')

for i in range(150):
  iris_data[i][0] = 1.0
#print(iris_data)

exp_data = iris_data[:99][:]
#print(exp_data)

import random
import math
import numpy as np



random.shuffle(exp_data)

train_x=[]
validation_x=[]
test_x=[]
train_y=[]
validation_y=[]
test_y=[]



for i in range(99):
  randomValue = random.uniform(0,1)
  if (randomValue  >= 0 and randomValue <=0.7):
    train_x.append(exp_data[i][:3])
    train_y.append(exp_data[i][4])
  elif (randomValue  > 0.7 and randomValue <=0.85):
    validation_x.append(exp_data[i][:3])
    validation_y.append(exp_data[i][4])
  else :
    test_x.append(exp_data[i][:3])
    test_y.append(exp_data[i][4])

import math
theta = np.array([10,20,35])
l_r = [0.1,0.01,0.001,0.0001,0.00001]
train_loss = []
train_loss_2 =[]
accuracy = []
for lr in l_r:
  for i in range (1000):
    tj = 0
    for s in range (len(train_x)):
      z = np.dot(train_x[s],theta)
      h = 1/(1 + np.exp(-z))
      #print(h)
      y = train_y[s]
      _h = 1-h
      a = (-y) * math.log(h)
      b = (1-y) * math.log(h)
      j = a - b
      tj = tj+j
      dv = [(train_x[s][0]*(h-y)),(train_x[s][1]*(h-y)),(train_x[s][2]*(h-y))]
      theta = [((theta[0]-dv[0])*lr),((theta[1]-dv[1])*lr),((theta[2]-dv[2])*lr)]

    tr_l = tj/s
    train_loss.append(tr_l)

  correct = 0
  for s in range (len(validation_x)):
    z = np.dot(train_x[s],theta)
    h = 1/(1 + np.exp(-z))
    #print(h)
    y = validation_y[s]
    if (h>=.5):
      h=1
    else:
      h=0
    if(h==y):
      correct += 1
  val_accuracy = correct/s
  accuracy.append(val_accuracy)

import matplotlib.pyplot as plt 

plt.plot(train_loss,1000,color = "green")
plt.xlabel('x - axis') 
plt.ylabel('y - axis')
plt.title('Loss vs Epoch') 
plt.show()

max_acc = max(accuracy)
max_index = number_list.index(max_acc)
accurate_lr = l_r[max_index]

for i in range (1000):
    tj = 0
    for s in range (len(train_x)):
      z = np.dot(train_x[s],theta)
      h = 1/(1 + np.exp(-z))
      #print(h)
      y = train_y[s]
      _h = 1-h
      a = (-y) * math.log(h)
      b = (1-y) * math.log(h)
      j = a - b
      tj = tj+j
      dv = [(train_x[s][0]*(h-y)),(train_x[s][1]*(h-y)),(train_x[s][2]*(h-y))]
      theta = [((theta[0]-dv[0])*accurate_lr),((theta[1]-dv[1])*accurate_lr),((theta[2]-dv[2])*accurate_lr)]

    tr_l = tj/s
    train_loss_2.append(tr_l)

import matplotlib.pyplot as plt 


plt.plot(train_loss_2,1000,color = "green")
plt.xlabel('x - axis') 
plt.ylabel('y - axis')
plt.title('Loss vs Epoch') 
plt.show()

correct = 0
for s in range (len(test_x)):
    z = np.dot(test_x[s],theta)
    h = 1/(1 + np.exp(-z))
    #print(h)
    y = test_y[s]
    if (h>=.5):
      h=1
    else:
      h=0
    if(h==y):
      correct += 1
test_accuracy = correct/s
print(test_accuracy)