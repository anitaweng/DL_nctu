import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math
import xlsxwriter
#################################################
#define softmax to prevent overflow
def softmax(x):
    max = np.max(np.add(x,1e-6))
    return np.divide(np.exp(np.add(x,1e-6) - max) , sum(np.exp(np.add(x,1e-6) - max)))
def sigmoid(x):
  return 1 / (1 + np.exp(np.add(-x,1e-6)))
#################################################
# load data, add bias
data = np.load('train.npz', allow_pickle = True)
x_train_pre1 = data['image'].reshape(12000, 784)
x_train = np.divide(x_train_pre1-np.min(x_train_pre1),np.max(x_train_pre1)-np.min(x_train_pre1))
y_train = data['label']
testdata = np.load('test.npz', allow_pickle = True)
x_test_pre = testdata['image'].reshape(5768, 784)
y_test_in = testdata['label']
x_test1 = np.divide(x_test_pre-np.min(x_test_pre),np.max(x_test_pre)-np.min(x_test_pre))
x_test = np.append(x_test1, np.ones((5768, 1)), axis=1)
hidden_layer_num = 1024
epochs = 500
eta = 0.001
error = []
rms = []
rms_test = []
mini_batch = 200
#################################################
# initial t
t = np.zeros((12000, 10))
for i in range(0,12000):
    t[i][int(y_train[i])] = 1
#################################################
# initial w
Wmap = np.random.rand(784+1, hidden_layer_num)
Wkm = np.random.rand(hidden_layer_num+1, 10)
y = np.zeros((12000,10))
result_y = np.zeros((12000,10))
#test
y_test = np.zeros((5768,10))
result_y_test = np.zeros((5768,10))
t_test = np.zeros((5768, 10))
for i in range(0,5768):
    t_test[i][int(y_test_in[i])] = 1
#################################################
for epoch in range(0,epochs):
    z = np.append(x_train, t, axis=1)
    np.random.shuffle(z)
    x_train = z[:, 0:784]
    t = z[:, 784:794]
    z_test = np.append(x_test, t_test, axis=1)
    np.random.shuffle(z_test)
    x_test = z_test[:, 0:785]
    t_test = z_test[:, 785:795]
    for batch in range(0, 12000, mini_batch):
        #train
        x = np.append(x_train[batch:batch+mini_batch, :], np.ones((mini_batch, 1)), axis=1) # mini_batch*785
        hidden = np.matmul(x, Wmap)
        hidden = sigmoid(hidden)
        hidden_bias = np.append(hidden, np.ones((mini_batch, 1)), axis =1)  # add bias #mini_batch*1025
        hidden2 = np.matmul(hidden_bias, Wkm) #mini_batch*10
        y[batch:batch+mini_batch, :] = np.zeros((mini_batch, 10))#mini_batch*10
        for i in range(0, mini_batch, 1):
            y[batch+i,:] = softmax(hidden2[i, :])
        # backpropagation for w,w update
        de_error2_hidden2 = np.subtract(y[batch:batch+mini_batch, :], t[batch:batch+mini_batch, :]) #mini_batch*10
        de_error2_Wkm = np.matmul(de_error2_hidden2.transpose(), hidden_bias) #10*1025
        de_error2_hidden = np.matmul(de_error2_hidden2, Wkm.transpose())  # mini_batch*1025
        de_error2_Wmap = np.matmul(de_error2_hidden.transpose(), x)  # 1025*785
        Wkm = np.subtract(Wkm, np.multiply(de_error2_Wkm.transpose(), eta)) #1025*10
        Wmap = np.subtract(Wmap, np.multiply(de_error2_Wmap[0:hidden_layer_num, :].transpose(), eta)) #785*1024
    error_matrix = np.multiply(t, np.log(np.add(y, 1e-6)))  # mini_batch*10
    error_matrix = np.multiply(error_matrix, -1/120000.0)  # mini_batch*10
    error.append(np.sum(error_matrix))
    rms.append(math.sqrt(np.sum(np.power((y - t),2))/120000.0))
    accuracy = 0
    for i in range (0,12000):
        result = np.where(y[i,:] == np.amax(y[i,:]))
        if t[i][result] == 1:
             accuracy = accuracy+1
    acc = accuracy/12000.0
    print("Epoch {:<5d} :Accuracy {:<8.3f} Loss {:<8.3f} Error_rate  {:<8.3f} ".format(epoch ,acc, error[-1], rms[-1]))
    #################################################
    a1 = np.matmul(x_test, Wmap)
    a11 = sigmoid(a1)
    a11_bias = np.append(a11, np.ones((5768, 1)), axis =1)  # add bias #mini_batch*1025
    a2 = np.matmul(a11_bias, Wkm) #mini_batch*10
    for i in range(0, 5768, 1):
        y_test[i,:] = softmax(a2[i, :])
    rms_test.append(math.sqrt(np.sum(np.power((y_test - t_test),2))/5768.0))

workbook = xlsxwriter.Workbook('confuse_matrix.xlsx')
worksheet = workbook.add_worksheet()
confuse_matrix = np.zeros((10,10))
for i in range (0,5768):
   confuse_matrix[np.argmax(t_test[i, :]),np.argmax(y_test[i, :])]=confuse_matrix[np.argmax(t_test[i, :]),np.argmax(y_test[i, :])]+1
row = 0
for col, data in enumerate(confuse_matrix):
    worksheet.write_column(row, col, data)
workbook.close()
    #################################################
plt.figure()
plt.plot(error)
plt.xlabel("Number of epochs")
plt.ylabel("Average cross entropy")
plt.title("Training loss")
plt.savefig("loss.png")
plt.show()
plt.close()

plt.figure()
plt.plot(rms)
plt.ylabel("Error rate")
plt.title("Train error rate")
plt.savefig("train_error.png")
plt.show()
plt.close()

plt.figure()
plt.plot(rms_test)
plt.ylabel("Error rate")
plt.title("Test error rate")
plt.savefig("test_error.png")
plt.show()
plt.close()