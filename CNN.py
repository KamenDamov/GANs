import torch

class CNN:
    input_height = None
    input_width = None
    input_channels = None
    kernel_size = None
    stride = 1
    padding = 0
    def __init__(self, input_height, input_width, input_channels, kernel_size) -> None:
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.kernel_size = kernel_size
    
    def get_activation_function(self, activation):
        if activation == 'relu':
            return torch.nn.ReLU()
        elif activation == 'sigmoid':
            return torch.nn.Sigmoid()
        else:
            raise ValueError('Invalid activation function')
    
    def produce_convolution_layer(self, input_channels, output_channels, kernel_size, stride, padding):
        return torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding)

    def produce_pooling_layer(self, kernel_size, stride, padding):
        return torch.nn.MaxPool2d(kernel_size, stride, padding)
    
    def produce_fully_connected_layer(self, input_size, output_size):
        return torch.nn.Linear(input_size, output_size)
    
    def produce_loss_function(self, loss_function):
        if loss_function == 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        else:    
            raise ValueError('Invalid loss function')
        
    def produce_optimizer(self, optimizer, learning_rate):
        if optimizer == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=learning_rate)
        elif optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=learning_rate)
        else:   
            raise ValueError('Invalid optimizer')
    
    def create_model(self, activation, loss_function, optimizer, learning_rate):
        activation_function = self.get_activation_function(activation)
        loss_function = self.produce_loss_function(loss_function)
        optimizer = self.produce_optimizer(optimizer, learning_rate)
        
        model = torch.nn.Sequential(
            self.produce_convolution_layer(self.input_channels, 32, self.kernel_size, self.stride, self.padding),
            activation_function,
            self.produce_pooling_layer(2, 2, 0),
            self.produce_convolution_layer(32, 64, self.kernel_size, self.stride, self.padding),
            activation_function,
            self.produce_pooling_layer(2, 2, 0),
            self.produce_convolution_layer(64, 128, self.kernel_size, self.stride, self.padding),
            activation_function,
            self.produce_pooling_layer(2, 2, 0),
            torch.nn.Flatten(),
            self.produce_fully_connected_layer(128, 10),
            activation_function,
            self.produce_fully_connected_layer(10, 10),
        )
        return model

