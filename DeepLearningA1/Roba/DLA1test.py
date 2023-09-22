import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import math
from collections import OrderedDict

#1. (10 pts) Complete the code


def create_dataset(w_star, x_range, sample_size, sigma, seed=None): 
    #creates a dataset x,y using the following parameters:
    #poly p(x) given by:
    #  w_star = coefficients
    #  x_range = domain of p(x) 
    #sample_size = how many points to generate
    #sigma = normal distibution standard deviation
    #seed = for random
    random_state = np.random.RandomState(seed)
    x = random_state.uniform(x_range[0] , x_range[1] , (sample_size)) 
    X = np.zeros((sample_size , w_star.shape[0]))
    
    for i in range(sample_size): 
        X[i, 0] = 1.
        for j in range(1, w_star.shape[0]):
             X[i, j] = x[i]**j    # completed

    y = X.dot(w_star)
    
    if sigma > 0 :
        y += random_state.normal(0.0 , sigma , sample_size)


    return X, x, y

#2. (10 pts) Use the completed code and the following parameters to generate
#    training and validation data points:

x_range=(-3,2)                  
w_star = np.array([-8,-4,2,1]) #-8 -4x -2xx - 1xxx

sigma = 0.5

#for #10.
#sigma = 2
#sigma = 4
#sigma = 8
#sigma = 100

seed_train=0
seed_test=1
sample_size=100
 
 #for 9.
divide=1   #1,2,10,20 -> 100,50,10,5

X_train,x_train,y_train= create_dataset(w_star, x_range, int(sample_size/divide), sigma, seed_train) #training
X_val,x_val,y_val= create_dataset(w_star, x_range, sample_size, sigma, seed_test) #validation

#11.
#w_star = np.array([-8,-4,2,1,1]) #extra

#3. (5 pts) Create a 2D scatterplot (using x and y) of the generated training and validation dataset.
plt.scatter(x_train, y_train, s=50, c='blue',alpha=0.5,label='training')
plt.scatter(x_val, y_val, s=50, c='orange',alpha=0.5,label='validation')
plt.show()

#TODO IN DOC
#4. (4 pts) Search for the documentation of torch.nn.Linear. Notice the flag bias. Explain what
#it does using equations. Should that flag set to True or False for this problem? Explain.

#https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
#bias – If set to False, the layer will not learn an additive bias. Default: True
#
#I dont think we need bias


#5. (20 pts) Adapt the linear regression in the 1D case presented in the lecture (at the end of Sec. 2.1) 
# to perform polynomial regression using the generated training dataset D′. 
# More specifically, find and report an estimate of w∗ = [−8, −4, 2, 1]T supposing that such a vector is unknown.

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model = nn.Linear(4, 1,False) # input dimension 4(features = poly coeffs), and output dimension 1 (predicted y). No Bias.

model = nn.Sequential(OrderedDict([ ('inp', nn.Linear(1, 4)),    ##in 4 features
                                    ('out', nn.Linear(4,1))])) #learn 5 coeffs of poly, out prediction?

#print(model.weight)
#print(model.bias)

model = model.to(DEVICE)
loss_fn = nn.MSELoss()

#6. (10 pts) Find and report a suitable learning rate 
#    and number of iterations for gradient descent.

#with 100 iter steps
#learning_rate = 0.1 #diverging
#learning_rate = 0.2 #diverging
#learning_rate = 0.3 #diverging
#learning_rate = 0.05 #diverging
#learning_rate = 0.05 #diverging
#learning_rate = 0.025 #diverging
learning_rate = 0.01  #very good ( minloss = 4)
#learning_rate = 0.005  #good ( minloss = 17)
#learning_rate = 0.001 #not so good ( minloss = 20)

#learning_rate = 0.0001 #better for sample=5?





optimizer = optim.SGD(model. parameters (), lr= learning_rate )


x_train=torch.from_numpy(x_train.reshape((int(sample_size/divide),1))).float().to(DEVICE)
x_val=torch.from_numpy(x_val.reshape((sample_size,1))).float().to(DEVICE)
y_train=torch.from_numpy(y_train.reshape((int(sample_size/divide),1))).float().to(DEVICE)
y_val=torch.from_numpy(y_val.reshape((100,1))).float().to(DEVICE)

#num_steps=100 #ok
#num_steps=1000 ##too much 
#num_steps=200 #(minLoss=1.2) #good? or overfitting? 
num_steps=1000 #(minLoss=0.2) #good? or overfitting? added early stop


####LOOP TRAIN EVALUATE
train_loss=[]
eval_loss=[]

##added rudimental early stoppping
#delta= 0.0001
delta=-999


for steps in range(num_steps):

    ##TRAINING
    model.train() #training mode
    optimizer.zero_grad() # init

    y_ = model(x_train) 

    loss= loss_fn(y_,y_train)
    print(f"Step {steps}: train loss: {loss}") 
    train_loss.append(loss.cpu().detach().numpy())


    loss.backward()
    optimizer.step()

    ##EVALUATION, predict using validation set
    model.eval()
    with torch.no_grad():
        y_ = model(x_val)
        val_loss = loss_fn(y_,y_val)
        
    print(f"Step {steps}: val loss: {val_loss}")
    eval_loss.append(val_loss.cpu().detach().numpy())
    
    ##added rudimental early stopping-> improvement<delta? stop
    if(math.isnan(eval_loss[len(eval_loss)-1]) or math.isinf(eval_loss[len(eval_loss)-1]) or (len(eval_loss)>2 and abs(eval_loss[len(eval_loss)-2] - eval_loss[len(eval_loss)-1]) <= delta)):
        
        print("##EARLY STOP %d/%d##" % (steps,num_steps))
        num_steps=steps+1
        break

########

#7. (10 pts) Plot the training and validation losses as a function of the gradient descent iterations.
plt.plot(range(0,num_steps), train_loss, c='blue',alpha=0.5)
plt.plot(range(0,num_steps), eval_loss, c='orange',alpha=0.5)
plt.show()

#8. (5 pts) Plot the polynomial defined by w∗ and the polynomial defined by your estimate wˆ .

def p(x,coeffs):
    #return - 8 - 4*x + 2*(x**2) + x**3 
    return coeffs[0] + coeffs[1]*x + coeffs[2]*(x**2) + coeffs[3]*(x**3)

#11 extra
def p4(x,coeffs):
    #return - 8 - 4*x + 2*(x**2) + x**3 + x**4
    return coeffs[0] + coeffs[1]*x + coeffs[2]*(x**2) + coeffs[3]*(x**3) + coeffs[4]*(x**4)
    
#print(model.weight)

#11.
#print(model.inp.weight)
print(model.out.weight)

#print(model.bias)
print(w_star)
#print(model.weight+model.bias)
print()

plt.plot(np.linspace(x_range[0],x_range[1],1000),[p(x,w_star) for x in np.linspace(x_range[0],x_range[1],1000)] , c='black')
#plt.plot(np.linspace(x_range[0],x_range[1],1000),[p4(x,model.weight.detach().numpy()[0]) for x in np.linspace(x_range[0],x_range[1],1000)] , c='red')

#11.
plt.plot(np.linspace(x_range[0],x_range[1],1000),[p(x,model.out.weight.detach().numpy()[0]) for x in np.linspace(x_range[0],x_range[1],1000)] , c='red')


plt.scatter(x_train, y_train, s=50, c='blue',alpha=0.5)
plt.scatter(x_val, y_val, s=50, c='orange',alpha=0.5)
plt.show()