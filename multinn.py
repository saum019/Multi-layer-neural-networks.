# Bhatt, Saumya

# %tensorflow_version 2.0.0
# %tensorflow_version 2.0.0
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss=None
         
    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        self.weights.append(tf.Variable(np.random.randn(self.input_dimension,num_nodes)))
        self.biases.append(tf.Variable(np.random.randn(num_nodes,1)))
        self.activations.append(transfer_function.lower())
        self.input_dimension=num_nodes      

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.biases[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number]=weights
        return self.weights[layer_number]

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number]=biases
        return self.biases[layer_number]

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
    
    def sigmoid(self, x):

        return tf.nn.sigmoid(x)

    def linear(self, x):
        return x

    def relu(self, x):
        out = tf.nn.relu(x)
        return out

   
    def predict(self, X):
        X_copy=X
        for values in range(len(self.weights)):
            predicted=(tf.matmul(X_copy,self.weights[values])+tf.transpose(self.biases[values]))
            if self.activations[values]=="sigmoid":
                trained=self.sigmoid(predicted)
            
            elif self.activations[values]=="linear":
                trained=self.linear(predicted)
                
            else:
                trained=self.relu(predicted)
            X_copy=trained
        
        return X_copy

    
    
    
    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        input_data=tf.data.Dataset.from_tensor_slices((X_train, y_train))
        input_data=input_data.batch(batch_size)
        for epoch in range(num_epochs):
         for count_step,(x,y) in enumerate(input_data):
            with tf.GradientTape(persistent=True) as data_grad_tape:
                predicted_y_val=self.predict(x)
                loss=self.calculate_loss(y,predicted_y_val)
                for weights_val in range(len(self.weights)):
                    partial_loss,partial_wb=data_grad_tape.gradient(loss,[self.weights[weights_val],self.biases[weights_val]])
                    self.weights[weights_val].assign_sub(alpha*partial_loss)
                    self.biases[weights_val].assign_sub(alpha*partial_wb)                  
                    
 

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        percent_error=0
        pred_y=self.predict(X)
        num_ofsamples,num_of_classes=np.shape(pred_y)
        target_class_predicted=np.argmax(pred_y,axis=1)
        commons=np.sum(np.where(target_class_predicted==y,0,1))
        percent_error=commons
        return percent_error/num_ofsamples
   



    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        final_predictions=[]
        predictions_Class=self.predict(X)
        for predictions in predictions_Class.numpy():  
            final_predictions.append(np.argmax(predictions))
        cmatrix=tf.math.confusion_matrix(y,final_predictions)
        return cmatrix

