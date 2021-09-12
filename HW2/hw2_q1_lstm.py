#create correlation
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch #
import torch.nn as nn#
import matplotlib.image as mpimg
from torch.autograd import Variable
import torch.nn.functional as F
import pygal
from pygal.maps.world import COUNTRIES
#import cairosvg
#import xlsxwriter
sns.set(font_scale=0.4)

with open('covid_19.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    a = []
    country_matrix = []
    country = []
    diff = [[0 for i in range(80)] for j in range(185)]
    coef_m = np.zeros((185,185))
    num = 0
    for row in rows:
        a.append(row)
        if num >= 3:
            country.append(row[0])
            ar = row[3:-1]
            for i in range(0, len(ar)):
                ar[i] = int(ar[i])
            country_matrix.append(ar)
        num = num + 1
    #print(len(country_matrix))
    #print(len(country_matrix[0])-1)
    for i in range(len(country_matrix)):
        #print(len(country_matrix[i]))
        for j in range(len(country_matrix[i])-1):
            #print(country_matrix[i][j+1])
            diff[i][j] = country_matrix[i][j+1] - country_matrix[i][j]
    for i in range(len(country_matrix)):
        for j in range(len(country_matrix)):
        #for j in range(len(diff[0])):
            #coef_v = np.corrcoef(country_matrix[i], country_matrix[j])
            coef_v = np.corrcoef(diff[i], diff[j])
            coef_m[i][j]=coef_v[0][1]
    co = pd.DataFrame(coef_m)
    df_lt = co.where(np.tril(np.ones(co.shape)).astype(np.bool)[0:185, 0:185])
    hmap = sns.heatmap(df_lt,cmap="Reds")
    hmap.figure.savefig("Correlation.png",format = 'png',dpi = 300)

    co1 = pd.DataFrame(coef_m[0:9, 0:9])
    df_lt1 = co1.where(np.tril(np.ones(co1.shape)).astype(np.bool)[0:9, 0:9])
    hmap1 = sns.heatmap(df_lt1, xticklabels=country[0:9], yticklabels=country[0:9], cmap="Reds")
    hmap1.figure.savefig("Correlation1.png", format='png', dpi=300)

    threshold = 0.5
    #threshold_matrix = coef_m[coef_m > threshold]
    set_C = []
    for i in range(185):
        for j in range(i,185):
            if coef_m[i][j] > threshold:
                #set_C.append((i,j))
                if not i in set_C:
                    set_C.append(i)
                if not j in set_C:
                    set_C.append(j)
    #print(set_C)
    #print(set_C[10][1])
    #print(len(set_C))
    '''workbook = xlsxwriter.Workbook('threshold_matrix.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    for col, data in enumerate(set_C):
        worksheet.write_column(row, col, str(data))
    workbook.close()'''
    '''file = open("threshold_matrix.txt", "w+")
    #file.write(str(threshold_matrix))
    file.close()'''


def window_data(data, window_size, set_C):
    x = []
    y = []

    for i in set_C:
        start_index = 0
        while (start_index + window_size) < len(data[i]):
            x.append(data[i][start_index: start_index + window_size])
            if data[i][start_index + window_size] > data[i][start_index + window_size - 1]:
                increase = [0,1]
            else:
                increase = [1,0]
            y.append(increase)
            start_index += 1
        assert len(x) == len(y)
    return x, y

    '''for i in set_C:
        start_index = 0
        while (start_index + window_size) < len(data[i]):
            increase = []
            x.append(data[i][start_index: start_index + window_size])
            for j in range(window_size-1):
                if data[i][start_index + j + 1] > data[i][start_index + j ]:
                    increase.append(1)
                else:
                    increase.append(0)
            y.append(increase)
            start_index += 1
        assert len(x) == len(y)
    return x, y'''



L = 35
num_dim = 2
window_size = L
x, y = window_data(diff, L, set_C)

'''embeds = nn.Embedding(window_size, num_dim)
#print(embeds)
input = torch.LongTensor(t)
#print(num_idx)
num_idx = Variable(input)
x = embeds(num_idx)'''

#print(x.size())
'''print(X)
print("\n")
print(y)
print(len(X))#13135 14245
print(len(y))'''

x_train = np.array(x[:7881])
y_train = np.array(y[:7881])
x_test = np.array(x[7881:])
y_test = np.array(y[7881:])

'''x_train = np.array(x[:8547])
y_train = np.array(y[:8547])
x_test = np.array(x[8547:])
y_test = np.array(y[8547:])'''
'''t_off = x_train - x_train.min()
t = (np.arange(t_off.max()+1) == t_off[...,None]).astype(int)'''

#t2 = pd.get_dummies(t1)
t_x = x_train.reshape((len(x_train), window_size, 1))
t_y = y_train.reshape((len(y_train), 2))

tt_x = x_test.reshape((len(x_test), window_size, 1))
tt_y = y_test.reshape((len(y_test), 2))

input_seq = torch.from_numpy(t_x)
target_seq = torch.Tensor(t_y)

tinput_seq = torch.from_numpy(tt_x)
ttarget_seq = torch.Tensor(tt_y)

'''input_seq = input_seq.cuda()
target_seq = target_seq.cuda()'''

'''len_x = len(x_train[0])
len_y = 1'''

#print(input_seq.size())
#print(target_seq.size())
#len_y = len(y_train[0])
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        #batch_size = 71

        batch_size = x.size(0)
        #print(batch_size)
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        c = self.init_hidden(batch_size)
        #print(x.size())
        #print(batch_size)
        #print(hidden.size())
        # Passing in the input and hidden state into the model and obtaining outputs

        '''is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.rnn.cuda()'''
        #print(hidden[0].size())
        out, hidden = self.lstm(x.float(), (hidden.float(),c.float()))
        #print(hidden.size())
        #out, hidden = self.rnn(x, hidden)

        #print(out.size())
        #print(hidden.size())
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        #print("\n")
        #print(out.size())
        return F.softmax(out, dim=1)
        #return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        #hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()
        return hidden

# Instantiate the model with hyperparameters

model = Model(input_size=1, output_size=2, hidden_dim=12, n_layers=1)
# We'll also set the model to the device that we defined earlier (default is CPU)
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False'''
'''is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
    #model.cuda()
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    #model.to(device)'''
device = torch.device("cpu")
model.to(device)

# Define hyperparameters
n_epochs = 500
lr=0.01
#batch_size = 71

# Define Loss, Optimizer
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

plot_loss = []
plot_acc = []

tplot_loss = []
tplot_acc = []
# Training Run
for epoch in range(1, n_epochs + 1):
    '''for i in range(0,len(x_test),batch_size):
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        input_seq.to(device)
        output, hidden = model(input_seq[i:i+batch_size])
        loss = criterion(output, target_seq[i:i+batch_size].view(-1).long())
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly'''

    optimizer.zero_grad()  # Clears existing gradients from previous epoch
    input_seq.to(device)
    output = model(input_seq)
    out_t = output.reshape(len(x_train),L,2)
    out_t_last = out_t[:,-1,:]
    prob, max = torch.max(out_t_last, 1)
    #print(out_t_last.size())
    #print(target_seq.reshape(7881, 1).size())
    #loss = criterion(out_t_last.reshape((7881,1)), target_seq.view(-1).long())
    #print(prob.size())
    #print(max.size())
    loss = criterion(out_t_last.reshape((len(x_train),2)), target_seq)
    loss.backward()  # Does backpropagation and calculates gradients
    optimizer.step()  # Updates the weights accordingly



    if epoch % 1 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs))
        print("Loss: {:.4f}".format(loss.item()))
        plot_loss.append(loss.item())
        #prediction = (out_t_last.reshape(7881, 1)>0.5).float()
        prediction = max
        #print(max)
        #print(torch.max(target_seq,1))
        val, ind = torch.max(target_seq, 1)
        correct = (prediction == ind).sum().float()
        total = len(target_seq)
        #acc_str = 'Accuracy: %f' % ((correct / total).cpu().detach().data.numpy())
        print("Accuracy: {:.4f}".format(correct / total))
        plot_acc.append(correct / total)

        tinput_seq.to(device)
        tout = model(tinput_seq)
        tout_t = tout.reshape(len(x_test), L, 2)
        tout_t_last = tout_t[:, -1, :]
        tprob, tmax = torch.max(tout_t_last, 1)
        tprediction = tmax
        tval, tind = torch.max(ttarget_seq,1)
        tcorrect = (tprediction == tind).sum().float()
        ttotal = len(ttarget_seq)
        #print(tprob)
        # acc_str = 'Accuracy: %f' % ((correct / total).cpu().detach().data.numpy())
        print("Test Accuracy: {:.4f}".format(tcorrect / ttotal))
        tplot_acc.append(tcorrect / ttotal)


plt.figure()
plt.plot(plot_loss)
plt.savefig("loss_train_lstm_"+str(L)+".png")
plt.figure()
plt.plot(plot_acc)
plt.savefig("acc_train_lstm_"+str(L)+".png")
plt.figure()
plt.plot(tplot_acc)
plt.savefig("acc_test_lstm_"+str(L)+".png")



def get_country_code(country_name):
    for code, name in COUNTRIES.items():
        if name == country_name:
            return code     # If the country wasn't found, return None.
        elif country_name == "Vietnam":
            return "vn"
        elif country_name == "Taiwan*":
            return "tw"
        elif country_name == "Venezuela":
            return "ve"
        elif country_name == "US":
            return "us"
        elif country_name == "Tanzania":
            return "tz"
        elif country_name == "Syria":
            return "sy"
        elif country_name == "South Sudan":
            return "sd"
        elif country_name == "Russia":
            return "ru"
        elif country_name == "North Macedonia":
            return "mk"
        elif country_name == "Moldova":
            return "md"
        elif country_name == "Libya":
            return "ly"
        elif country_name == "Laos":
            return "la"
        elif country_name == "Korea, South":
            return "kr"
        elif country_name == "Iran":
            return "ir"
        elif country_name == "Holy See":
            return "va"
        elif country_name == "Dominica":
            return "do"
        elif country_name == "Czechia":
            return "cz"
        elif country_name == "Congo (Kinshasa)":
            return "cd"
        elif country_name == "Congo (Brazzaville)":
            return "cg"
        elif country_name == "Cabo Verde":
            return "cv"
        elif country_name == "Burma":
            return "mn"
        elif country_name == "Brunei":
            return "bn"
        elif country_name == "Bolivia":
            return "bo"

    return None

worldmap_chart = pygal.maps.world.World()
worldmap_chart.title = 'Covid-19 the number of confirmed people will increase or decrease in the next day'
diff = np.array(diff)
plot_x = diff[:,-1*L:].reshape((len(diff), window_size, 1))
plot_input_seq = torch.from_numpy(plot_x)
#print(plot_input_seq.size())
plot_country = []
'''for i in range(len(diff)):
    plot_input_seq[i,:,:].to(device)
    plot_out = model(plot_input_seq[i,:,:].reshape(1,window_size, 1))
    print(plot_out.size())
    plot_out_last = plot_out[:, -1, :]
    plot_prob, plot_max = torch.max(plot_out_last, 1)
    plot_country.append((country[i],plot_max,plot_prob))'''


plot_input_seq.to(device)
plot_out = model(plot_input_seq)
plot_out_t = plot_out.reshape(len(diff), L, 2)
plot_out_last = plot_out_t[:, -1, :]
#print(plot_out_last.size())
plot_prob, plot_max = torch.max(plot_out_last, 1)
for i in range(len(diff)):
    plot_country.append((country[i],plot_max.tolist()[i],plot_prob.tolist()[i]))

#print(plot_country)
#print(plot_country)
decre = {}
incre = {}
for i in range(len(diff)):
    if plot_country[i][1] == 0:
        decre.update({get_country_code(plot_country[i][0]) : plot_country[i][2]})
    else:
        incre.update({get_country_code(plot_country[i][0]) : plot_country[i][2]})


worldmap_chart.add('Increase', incre)
worldmap_chart.add('Decrease', decre)
#worldmap_chart.render()
worldmap_chart.render_to_file('world_map_'+str(L)+'.svg')
#worldmap_chart.render_to_png(filename = 'world_map.png')

# This function takes in the model and character as arguments and returns the next character prediction and hidden state
'''def predict(model, num):
    # One-hot encoding our input to fit into the model
    num = np.array([[char2int[c] for c in num]])
    num = one_hot_encode(num, dict_size, num.shape[1], 1)
    num = torch.from_numpy(num)
    num.to(device)

    out, hidden = model(num)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    num_ind = torch.max(prob, dim=0)[1].item()

    return int2char[num_ind], hidden'''

# This function takes the desired output length and input characters as arguments, returning the produced sentence
'''def sample(model, out_len, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)'''



