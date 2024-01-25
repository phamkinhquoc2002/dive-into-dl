import torch
import numpy as numpy
import random
from torch import nn
from d2l import torch as d2l



class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        """
        Abstract method to save hyperparameters.

        Parameters:
        - ignore (list): List of hyperparameter names to ignore while saving.
        
        Raises:
        - NotImplementedError: This method needs to be implemented by subclasses.
        """
        raise NotImplementedError


    
class ProgressBoard(HyperParameters):
    def __init__(self, xlabel=None, ylabel=None,
                 xlim=None, ylim=None,
                 xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        """
        Constructor for the ProgressBoard class.

        Parameters:
        - xlabel (str): Label for the x-axis.
        - ylabel (str): Label for the y-axis.
        - xlim (tuple): Tuple representing the x-axis limits.
        - ylim (tuple): Tuple representing the y-axis limits.
        - xscale (str): Scale of the x-axis ('linear' by default).
        - yscale (str): Scale of the y-axis ('linear' by default).
        - ls (list): List of line styles for plotting.
        - colors (list): List of colors for plotting.
        - fig (matplotlib.figure.Figure): Matplotlib figure for plotting.
        - axes (matplotlib.axes.Axes): Matplotlib axes for plotting.
        - figsize (tuple): Tuple representing the figure size.
        - display (bool): Flag to determine whether to display the plot.
        """
        self.save_parameters()

    def draw(self, x, y, label, every_n=1):
        """
        Method to draw a plot with specified data.

        Parameters:
        - x: Data for the x-axis.
        - y: Data for the y-axis.
        - label (str): Label for the plot.
        - every_n (int): Draw a point every n data points.
        
        Raises:
        - NotADirectoryError: This method needs to be implemented.
        """
        raise NotADirectoryError



class Module(nn.Module, d2l.HyperParameters):
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        """
        Constructor for the Module class.

        Parameters:
        - plot_train_per_epoch (int): Number of times to plot training statistics per epoch.
        - plot_valid_per_epoch (int): Number of times to plot validation statistics per epoch.
        """
        super().__init__()
        self.save_parameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        """
        Abstract method to calculate the loss.

        Parameters:
        - y_hat: Predicted values.
        - y: Actual values.

        Raises:
        - NotImplementedError: This method needs to be implemented by subclasses.
        """
        raise NotImplementedError()

    def forward(self, X):
        """
        Forward pass of the neural network.

        Parameters:
        - X: Input data.

        Raises:
        - NotImplementedError: This method needs to be implemented by subclasses.
        """
        assert hasattr(self, 'net')
        raise NotImplementedError()

    def plot(self, key, value, train):
        """
        Method to plot training or validation statistics.

        Parameters:
        - key (str): Key for the statistic (e.g., 'loss').
        - value: Value of the statistic.
        - train (bool): Flag indicating whether it is a training statistic.

        Raises:
        - AssertionError: Raised if the Module has no 'trainer' attribute.
        - AssertionError: Raised if the Module has no 'board' attribute.
        - NotADirectoryError: Raised if the 'draw' method of 'board' is not implemented.
        """
        assert hasattr(self, 'trainer')
        assert hasattr(self, 'board')
        self.board.xlabel = 'epoch'

        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_valid_batches / self.plot_valid_per_epoch

        self.board.draw(x, value.to(d2l.gpu()).detach().numpy(),
                        ('train_' if train else 'val_') + key, every_n=int(n))

    def training_step(self, batch):
        """
        Training step of the module.

        Parameters:
        - batch: Training batch.

        Returns:
        - l: Loss value.

        Raises:
        - NotImplementedError: This method needs to be implemented by subclasses.
        """
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        """
        Validation step of the module.

        Parameters:
        - batch: Validation batch.

        Raises:
        - NotImplementedError: This method needs to be implemented by subclasses.
        """
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        """
        Abstract method to configure optimizers.

        Raises:
        - NotImplementedError: This method needs to be implemented by subclasses.
        """
        raise NotImplementedError



class DataModule(d2l.HyperParameters):
    def __init(self, root='../data'):
        """
        Constructor for the DataModule class.

        Parameters:
        - root (str): Root directory for data (default is '../data').
        """
        self.save_hyperparameters()

    def get_dataloader(self, train):
        """
        Abstract method to get a DataLoader.

        Parameters:
        - train (bool): Flag indicating whether to get a training DataLoader.

        Raises:
        - NotImplementedError: This method needs to be implemented by subclasses.
        """
        raise NotImplementedError()

    def train_loader(self):
        """
        Convenience method to get a training DataLoader.

        Returns:
        - DataLoader: Training DataLoader.
        
        Raises:
        - NotImplementedError: This method needs to be implemented by subclasses.
        """
        return self.get_dataloader(train=True)

    def val_loader(self):
        """
        Convenience method to get a validation DataLoader.

        Returns:
        - DataLoader: Validation DataLoader.
        
        Raises:
        - NotImplementedError: This method needs to be implemented by subclasses.
        """
        return self.get_dataloader(train=False)



class Trainer(d2l.HyperParameters):
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        """
        Constructor for the Trainer class.

        Parameters:
        - max_epochs (int): Maximum number of training epochs.
        - num_gpus (int): Number of GPUs (default is 0).
        - gradient_clip_val (int): Value for gradient clipping (default is 0).
        """
        self.save_hyperparameters()
        assert num_gpus == 0, "No GPUs supported yet"

    def prepare_data(self, data):
        """
        Method to prepare data for training.

        Parameters:
        - data: DataModule instance.

        Raises:
        - AttributeError: Raised if 'train_data' or 'val_data' attributes are not found in the DataModule.
        """
        self.train_dataloader = data.train_loader()
        self.val_dataloader = data.val_loader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = len(self.val_dataloader) if len(self.val_dataloader) is not None else 0

    def prepare_model(self, model):
        """
        Method to prepare the model for training.

        Parameters:
        - model: Instance of the model to be trained.

        Raises:
        - AttributeError: Raised if the 'trainer' or 'board' attributes are not found in the model.
        """
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        """
        Method to initiate the training process.

        Parameters:
        - model: Instance of the model to be trained.
        - data: DataModule instance.
        """
        self.prepare_data(data)
        self.prepare_model(model)
        self.opti = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0

        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        """
        Abstract method representing a single training epoch.

        Raises:
        - NotImplementedError: This method needs to be implemented by subclasses.
        """
        raise NotImplementedError()



class SGD(d2l.HyperParameters):
    def __init__(self, params, lr):
        """
        Constructor for the SGD class.

        Parameters:
        - params (iterable): Iterable of parameters to optimize.
        - lr (float): Learning rate for the optimizer.
        """
        self.save_hyperparameters()

    def step(self):
        """
        Method to perform a single optimization step.

        - Updates each parameter based on the learning rate and gradient.
        
        Raises:
        - AttributeError: Raised if 'params' attribute is not found in the SGD instance.
        - AttributeError: Raised if 'lr' attribute is not found in the SGD instance.
        """
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        """
        Method to zero out the gradients of all parameters.

        Raises:
        - AttributeError: Raised if 'params' attribute is not found in the SGD instance.
        """
        for param in self.params:
            if param is not None:
                param.grad.zero_()



class Classifier(d2l.Module):
    def validation_step(self, batch):
        """
        Validation step for the Classifier.

        Parameters:
        - batch: Validation batch.

        Raises:
        - AttributeError: Raised if 'plot' method is not found in the Classifier.
        - AttributeError: Raised if 'loss' method is not found in the Classifier.
        - AttributeError: Raised if 'accuracy' method is not found in the Classifier.
        """
        Y_hat = self(*batch[:-1])

        # Plot validation loss
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)

        # Plot validation accuracy
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)



@d2l.add_to_class(d2l.DataModule)
def tensorLoader(self, tensors, train, indices=slice(0, None)):
    """
    Method to create a DataLoader for a set of tensors.

    Parameters:
    - tensors (tuple): Tuple of tensors to be loaded.
    - train (bool): Flag indicating whether it is a training DataLoader.
    - indices (slice): Slice specifying the indices to include (default is slice(0, None)).

    Returns:
    - torch.utils.data.DataLoader: DataLoader for the specified tensors.
    """
    tensors = tuple(a[indices] for a in tensors)
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)



@d2l.add_to_class(d2l.Trainer)
def fit_epoch(self):
    """
    Method representing a single training epoch.

    - Sets the model to train mode.
    - Iterates over the training DataLoader, updates the model parameters, and calculates the training loss.
    - Performs gradient clipping if the gradient_clip_val is greater than 0.
    - Sets the model to evaluation mode if there is a validation DataLoader and performs validation steps.

    Raises:
    - AttributeError: Raised if 'train_dataloader' attribute is not found in the Trainer.
    - AttributeError: Raised if 'val_dataloader' attribute is not found in the Trainer.
    """
    self.model.train()

    # Training Phase
    for batch in self.train_dataloader:
        self.train_batch_idx += 1
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()

        with torch.no_grad():
            loss.backward()

            if self.gradient_clip_val > 0:
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()

    # Validation Phase
    if self.val_dataloader is not None:
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1



@d2l.add_to_class(d2l.Module)
def configure_optimizers(self):
    """
    Method to configure optimizers for the Module.

    Returns:
    - torch.optim.SGD: Stochastic Gradient Descent optimizer with the model parameters and specified learning rate.
    
    Raises:
    - AttributeError: Raised if 'parameters' attribute is not found in the Module.
    - AttributeError: Raised if 'lr' attribute is not found in the Module.
    """
    return torch.optim.SGD(self.parameters(), lr=self.lr)



@d2l.add_to_class(Classifier)
def accuracy(self, Y_hat, Y, averaged=True):
    """
    Method to calculate the accuracy of predictions.

    Parameters:
    - Y_hat: Predicted values.
    - Y: Actual values.
    - averaged (bool): Flag indicating whether to return the averaged accuracy (default is True).

    Returns:
    - torch.Tensor: Accuracy value.

    Raises:
    - AttributeError: Raised if 'dtype' attribute is not found in the Classifier.
    """
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    preds = Y_hat.argmax(axis=1).type(Y.dtype)
    compare = (preds == Y.reshape(-1)).type(torch.float32)
    return compare.mean() if averaged else compare



@d2l.add_to_class(d2l.Classifier)
def layer_summary(self, X_shape):
    """
    Generate a summary of the layer output shapes in the Classifier's network.

    Parameters:
    - X_shape (tuple): The input shape (excluding batch size) to the network.

    Prints:
    - Output shape after passing through each layer in the network.

    Note:
    - This function does not modify the model or perform an actual forward pass.
      It only prints the output shapes based on a randomly generated input.

    Example usage:
    ```
    classifier_instance.layer_summary((3, 32, 32))  # Assuming input shape is (3, 32, 32)
    ```

    """
    # Generate a random input tensor with the specified shape
    X = torch.randn(*X_shape)

    # Iterate through each layer in the network and print output shapes
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, "output_shape:\t", X.shape)



@d2l.add_to_class(d2l.Trainer)
def prepare_batch(self, batch):
  return batch



@d2l.add_to_class(d2l.DataModule)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        batch_indices = torch.tensor(indices[i: i+self.batch_size])
        yield self.X[batch_indices], self.y[batch_indices]

  
