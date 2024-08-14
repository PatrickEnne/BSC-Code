import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xlrd 


class Layer():
    

    def __init__(self, input_n, output_n,activationFunction,dropout_rate=0.0) :
       
       #initialize weights
        if activationFunction=="relu":
            self.weights=self.he_Initialization(input_n, output_n)
        if activationFunction=="sigm":
            self.weights=self.xavier_Initialization(input_n=input_n,output_n=output_n)

        #initialize biases
        self.biases=np.zeros((1,output_n))


        self.activation=activationFunction
        self.dropout_rate=dropout_rate
        self.dropout_mask = None

        #initialize Adam optimizer
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)
        self.t = 0


    
    def xavier_Initialization(self,input_n,output_n):
        
        W = np.random.randn(input_n, output_n) * np.sqrt(1.0 / input_n)

        return W



    #weights are being initialized with the He-Initialization to accomodate for the ReLu activation function
    def he_Initialization(self,input_n, output_n):


        #calc range for the weights
        std = np.sqrt(2.0 / input_n)

        # calculate random weight
        weight=np.random.randn(input_n,output_n)*std

        return weight
    

    #ReLu: rectified linear activation unit used for hidden layers
    def relu(self,x):
        return np.maximum(x,0)


    def relu_derivative(self,x):
       return np.where(x > 0, 1, 0)
    
    # Leaky ReLU
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)
    


    #Sigmoid activation function for output layer
    def sigm(self,x):
        
        #x = np.clip(x, -10, 10)  
        return 1/(1 + np.exp(-x))
    
    
    def sigm_derivative(self,output):
        return output*(1-output)
    
    

    def forwardPropagation(self, input_data,training=True):

        self.inputdata=input_data
        self.x=np.dot(self.inputdata,self.weights)
        self.x+=self.biases

        if(self.activation=="relu"):
            output=self.relu(self.x)
        elif(self.activation=="sigm"):
            output=self.sigm(self.x)
        else:
            print("wrong activation function!")
        if training and self.dropout_rate > 0.0:
            self.dropout_mask = (np.random.rand(*output.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            output *= self.dropout_mask
        
        self.output=output
            
        return output    



    def backwardPropagation(self, delta_val,learningrate):

        
        
        if self.activation=="relu":
            self.delta=delta_val*self.relu_derivative(self.output)
        elif self.activation=="sigm":
            self.delta=delta_val*self.sigm_derivative(self.output)
        else:
            print("wrong activation function!")

        if self.dropout_rate > 0.0:
            self.delta *= self.dropout_mask
        
        m=self.inputdata.shape[1]
        
        weights_gradient = np.dot(self.inputdata.T, self.delta)/m
        biases_gradient = np.sum(self.delta, axis=0, keepdims=True)/m
        self.delta= np.dot(self.delta,self.weights.T)

        
        weights_gradient,biases_gradient=self.clip_gradients(weights_gradient=weights_gradient,biases_gradient=biases_gradient)
        #self.update_weights_biases(weights_gradient,biases_gradient=biases_gradient,learningrate=learningrate)
        self.adam_optimizer(weights_gradient=weights_gradient,biases_gradient=biases_gradient,learning_rate=learningrate,beta1=0.9,beta2=0.999,epsilon=1e-8)
        #print("weights after optimization:",self.weights)

        #print("weights-gradient after clipping:")
        #print(weights_gradient)
        return self.delta
    

    def clip_gradients(self,weights_gradient, biases_gradient, threshold=1.0):
        #weights_norm = np.linalg.norm(weights_gradient)
        #biases_norm = np.linalg.norm(biases_gradient)
    
        #if weights_norm > threshold:
        weights_gradient = np.clip(weights_gradient,-threshold,threshold)
            #(weights_gradient / weights_norm) * threshold
        #if biases_norm > threshold:
        biases_gradient = np.clip(biases_gradient,-threshold,threshold)
            #(biases_gradient / biases_norm) * threshold
    
        return weights_gradient, biases_gradient

    '''
    def update_weights_biases(self,weights_gradient,biases_gradient, learningrate,lambd=0.01):

        self.weights-=learningrate*weights_gradient
        self.biases-=learningrate*biases_gradient
    '''    

    def adam_optimizer(self, weights_gradient, biases_gradient, learning_rate, beta1, beta2, epsilon, lambd=0.01):
        self.t += 1
    
        # Update biased first moment estimate
        self.m_weights = beta1 * self.m_weights + (1 - beta1) * weights_gradient
        self.m_biases = beta1 * self.m_biases + (1 - beta1) * biases_gradient
    
    # Update biased second raw moment estimate
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * (weights_gradient ** 2)
        self.v_biases = beta2 * self.v_biases + (1 - beta2) * (biases_gradient ** 2)
    
    # Compute bias-corrected first moment estimate
        m_hat_weights = self.m_weights / (1 - beta1 ** self.t)
        m_hat_biases = self.m_biases / (1 - beta1 ** self.t)
    
    # Compute bias-corrected second raw moment estimate
        v_hat_weights = self.v_weights / (1 - beta2 ** self.t)
        v_hat_biases = self.v_biases / (1 - beta2 ** self.t)
    
    # Update weights with L2 regularization
        self.weights -= learning_rate * (m_hat_weights / (np.sqrt(v_hat_weights) + epsilon) + lambd * self.weights)
        self.biases -= learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + epsilon)