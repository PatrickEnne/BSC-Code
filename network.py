from layers import Layer
import numpy as np
class Network:
    best_val_loss = float('inf')
    best_weights = None
 
    def __init__(self, input_len, output_len, dropout_rates):
        self.layers = []
        self.build_network(input_len, output_len, dropout_rates)
    
    def build_network(self, input_len, output_len, dropout_rates):
        # Validate inputs
        if len(output_len) != len(dropout_rates):
            raise ValueError("Length of output_len and dropout_rates must be the same")

        # Add the first layer
        self.layers.append(Layer(input_n=input_len, output_n=output_len[0], activationFunction="relu", dropout_rate=dropout_rates[0]))

         # Add subsequent layers
        for i in range(1, len(output_len)):
            activation_function = "relu" if i < len(output_len) - 1 else "sigm"
            dropout_rate = dropout_rates[i]
            self.layers.append(Layer(input_n=output_len[i-1], output_n=output_len[i], activationFunction=activation_function, dropout_rate=dropout_rate))
        
    def forwardPropagation(self, x):
        # Pass the input through each layer
        for layer in self.layers:
            x = layer.forwardPropagation(x)
        return x
  
    #mean squared
    #def compute_loss(self, Y, output):
     #   return np.mean((Y - output) ** 2)
    
    def get_all_weights(self):
        # Collect weights from all layers
        return [layer.get_weights() for layer in self.layers]

    def set_all_weights(self, best_weights):
    # Ensure that best_weights is a list and matches the number of layers
        if not isinstance(best_weights, list):
            raise TypeError("best_weights must be a list")
        if len(best_weights) != len(self.layers):
            raise ValueError("Number of weight sets does not match number of layers")

    # Set weights for each layer
        for layer, weights in zip(self.layers, best_weights):
            if hasattr(layer, 'set_weights'):
                layer.set_weights(weights)
            else:
                raise AttributeError("Layer does not have a 'set_weights' method")
        
    #cross-entropy
    def compute_loss(self, Y, output):
        epsilon = 1e-15  # To prevent log(0)
        output = np.clip(output, epsilon, 1 - epsilon)  # Clip the output to avoid log(0)
        loss = -np.mean(Y * np.log(output) + (1 - Y) * np.log(1 - output))

        loss = -np.mean(Y * np.log(output-epsilon))
        if(loss<self.best_val_loss):
            self.best_val_loss=loss
            self.best_weights=self.get_all_weights()

        return loss
        
    
    def backwardPropagation(self, error, learning_rate):
        delta = error
        for layer in reversed(self.layers):
            delta = layer.backwardPropagation(delta, learningrate=learning_rate)

    def adjust_learning_rate(self,  initial_lr, decay_factor=0.5):
        lr = decay_factor*initial_lr
        print(lr)
        return lr
    

    def train_network(self,learning_rate,X_train,Y_train,epoch_n,batch_size):
        losses = []
        
        decay_factor = 0.5
        decay_step = 20
        for epoch in range(epoch_n):
            if epoch%decay_step==0 and epoch!=0:
                learning_rate= self.adjust_learning_rate(  learning_rate, decay_factor)

            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            Y_train_shuffled = Y_train[indices]
            epoch_loss = 0
            num_processed_samples = 0
            for start in range(0, X_train.shape[0], batch_size):
                end = min(start + batch_size, X_train.shape[0])
                X_batch = X_train_shuffled[start:end]
                Y_batch = Y_train_shuffled[start:end]
            
            #forward propagation
                output=self.forwardPropagation(X_batch)
                loss = self.compute_loss(Y_batch, output)
                epoch_loss += loss * len(X_batch)  
                num_processed_samples += len(X_batch)
            
            #backpropagation
                #error of outputlayer
                error=-(np.divide(Y_batch, output) - np.divide(1 - Y_batch, 1 - output))
                self.backwardPropagation(error=error,learning_rate=learning_rate)
                
            average_loss = epoch_loss / num_processed_samples
            losses.append(average_loss)
        print(f"Type of best_weights: {type(self.best_weights)}")
        print(f"Contents of best_weights: {self.best_weights}")    
        self.set_all_weights(self.best_weights)
        return losses