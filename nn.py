import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
#from keras.optimizers import sgd,rmsprop,adagrad



class CNN(object):
    def __init__(self):
        """
        Initialize multi-layer neural network

        """
        self.layer=[]
        self.model=keras.Model()
        self.loss=None
        self.optimizer=None
        self.metric=[]


    def add_input_layer(self, shape=(2,),name="" ):
        """
         This function adds an input layer to the neural network. If an input layer exist, then this function
         should replcae it with the new input layer.
         :param shape: input shape (tuple)
         :param name: Layer name (string)
         :return: None
         """
        ip = tf.keras.Input(shape=shape, name=name)
        self.model = tf.keras.Model(ip, ip)



    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: Number of nodes
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid",
         "Softmax"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: None
         """
        dense_layer=tf.keras.layers.Dense(num_nodes, activation=activation, name=name, trainable=trainable)(self.model.output)
        self.model=tf.keras.Model(self.model.input, dense_layer)
        
        
    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,
                         activation="Relu",name="",trainable=True):
        """
         This function adds a conv2d layer to the neural network
         :param num_of_filters: Number of nodes
         :param num_nodes: Number of nodes
         :param kernel_size: Kernel size (assume that the kernel has the same horizontal and vertical size)
         :param padding: "same", "Valid"
         :param strides: strides
         :param activation: Activation function for the layer. Possible values are "Linear", "Relu", "Sigmoid"
         :param name: Layer name (string)
         :param trainable: Boolean
         :return: Layer object
         """
        conv2d_layer=tf.keras.layers.Conv2D(num_of_filters, kernel_size, strides=strides, padding=padding, activation=activation, name=name, trainable=trainable)(self.model.output)
        #self.layer.append(conv2d_layer(self.layer[-1]))
        self.model=tf.keras.Model(self.model.input, conv2d_layer)
        return conv2d_layer
        
    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):
        """
         This function adds a maxpool2d layer to the neural network
         :param pool_size: Pool size (assume that the pool has the same horizontal and vertical size)
         :param padding: "same", "valid"
         :param strides: strides
         :param name: Layer name (string)
         :return: Layer object
         """
        pool2d_layer=tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding=padding, strides=strides, name=name)(self.model.output)
        self.model=tf.keras.Model(self.model.input, pool2d_layer)
        return pool2d_layer
        
        
    def append_flatten_layer(self,name=""):
        """
         This function adds a flattening layer to the neural network
         :param name: Layer name (string)
         :return: Layer object
         """
        flatten_layer=tf.keras.layers.Flatten(name=name)(self.model.output)
        self.model=tf.keras.Model(self.model.input, flatten_layer)
        return flatten_layer
        
    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):
        """
        This function sets the trainable flag for a given layer
        :param layer_number: an integer or a list of numbers.Layer numbers start from layer 0.
        :param layer_names: a string or a list of strings (if both layer_number and layer_name are specified, layer number takes precedence).
        :param trainable_flag: Set trainable flag
        :return: None
        """
        if layer_numbers is None:
            for n in layer_names:
                self.model.get_layer(name=n).trainable=trainable_flag
        else:
            for l in layer_numbers:
                self.model.get_layer(index=l).trainable=trainable_flag
            
                

    def get_weights_without_biases(self,layer_number=None,layer_name=""):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: Weight matrix for the given layer (not including the biases). If the given layer does not have
          weights then None should be returned.
         """
        if len(self.model.get_layer(name=layer_name, index=layer_number).weights)==0:
            return None
        else:
            return self.model.get_layer(name=layer_name, index=layer_number).weights[0].numpy()
            


    def get_biases(self,layer_number=None,layer_name=""):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: biases for the given layer (If the given layer does not have bias then None should be returned)
         """
        if len(self.model.get_layer(name=layer_name, index=layer_number).weights)==0:
            return None
        else:
            return self.model.get_layer(name=layer_name, index=layer_number).weights[1].numpy()
        

    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
         :return: None
         """
        w=[]
        w.append(weights)
        w.append(self.get_biases(layer_number=layer_number, layer_name=layer_name))
        self.model.get_layer(name=layer_name, index=layer_number).set_weights(w)
    def set_biases(self,biases,layer_number=None,layer_name=""):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :param layer_name: Layer name (if both layer_number and layer_name are specified, layer number takes precedence).
        :return: none
        """
        b=[]
        b.append(self.get_weights_without_biases(layer_number=layer_number, layer_name=layer_name))
        b.append(biases)
        self.model.get_layer(name=layer_name, index=layer_number).set_weights(b)
    def remove_last_layer(self):
        """
        This function removes a layer from the model.
        :return: removed layer
        """
        removed_layer=self.model.layers[-1]
        self.model=keras.Model(self.model.input, self.model.layers[-2].output)
        return removed_layer

    def load_a_model(self,model_name="",model_file_name=""):
        """
        This function loads a model architecture and weights.
        :param model_name: Name of the model to load. model_name should be one of the following:
        "VGG16", "VGG19"
        :param model_file_name: Name of the file to load the model (if both madel_name and
         model_file_name are specified, model_name takes precedence).
        :return: model
        """
        if model_name=='VGG16':
            self.model=keras.applications.VGG16()
            return self.model
        elif model_name=='VGG19':
            self.model=keras.applications.VGG19()
            return self.model
        elif model_file_name!=None:
            self.model=keras.models.load_model(model_file_name)
            return self.model
    def save_model(self,model_file_name=""):
        """
        This function saves the current model architecture and weights together in a HDF5 file.
        :param file_name: Name of file to save the model.
        :return: model
        """
        return self.model.save(model_file_name)


    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        """
        This function sets the loss function.
        :param loss: loss is a string with the following choices:
        "SparseCategoricalCrossentropy",  "MeanSquaredError", "hinge".
        :return: none
        """
        self.loss=loss

    def set_metric(self,metric):
        """
        This function sets the metric.
        :param metric: metric should be one of the following strings:
        "accuracy", "mse".
        :return: none
        """
        self.metric.append(metric)

    def set_optimizer(self,optimizer="RMSprop",learning_rate=0.01,momentum=0.0):
        """
        This function sets the optimizer.
        :param optimizer: Should be one of the following:
        "SGD" , "RMSprop" , "Adagrad" ,
        :param learning_rate: Learning rate
        :param momentum: Momentum
        :return: none
        """
        if optimizer=="SGD":
            self.optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer=="RMSprop":
            self.optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum)
        elif optimizer=="Adagrad":
            self.optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate, momentum=momentum)

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Input tensor.
        :return: Output tensor.
        """
        Xnew=X
        return self.model.predict(Xnew)
        

    def evaluate(self,X,y):
        """
         Given array of inputs and desired ouputs, this function returns the loss value and metrics of the model.
         :param X: Array of input
         :param y: Array of desired (target) outputs
         :return: loss value and metric value
         """
        return self.model.evaluate(X, y)
       
    def train(self, X_train, y_train, batch_size, num_epochs):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X_train: Array of input
         :param y_train: Array of desired (target) outputs
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :return: list of loss values. Each element of the list should be the value of loss after each epoch.
         """
        self.model.compile(loss= self.loss, optimizer=self.optimizer, metrics= self.metric)
        loss_vals=self.model.fit(X_train, y_train, batch_size, num_epochs)
        return loss_vals

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255


    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    my_cnn = CNN()
    my_cnn.add_input_layer(shape=(28,28,1))
    my_cnn.append_conv2d_layer(32, (3,3), padding='same',activation='relu', name="conv1", trainable=True)
    my_cnn.append_maxpooling2d_layer(pool_size=(3,3), padding="same", strides=2, name="pool")
    my_cnn.append_conv2d_layer(16, (3,3), padding='same',activation='relu', name="conv2", trainable=True)
    my_cnn.append_flatten_layer(name="flatten")
    my_cnn.append_dense_layer(num_nodes=1000, activation="relu", name="dense1", trainable=True)
    my_cnn.append_dense_layer(num_nodes=10, activation="softmax", name="dense2", trainable=True)
    my_cnn.set_loss_function(loss="categorical_crossentropy")
    my_cnn.set_optimizer(optimizer="RMSprop", learning_rate=0.01, momentum=0.0)
    my_cnn.set_metric(['accuracy'])
    print('\n***Train data***')
    my_cnn.train(x_train, y_train,batch_size=128,num_epochs=1)
    print('\n*** Evaluate on test data***')
    results = my_cnn.evaluate(x_test, y_test)
    print('test loss, test acc:', results)