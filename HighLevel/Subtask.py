import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Subtask:
    def __init__(self, model, dataset, trainer, train_params, initial_importance, sampling_time):
        self.model = model
        self.data = dataset 
        self.trainer = trainer
        self.train_params = train_params
        self.importance = initial_importance
        self.sampling_time = sampling_time
        self.history = {} # batch_level

    def __lt__(self, other):
        return self.importance > other.importance # min heap
    
    def update_dataset(self):
        pass


class SubtaskData(Dataset):
    def __init__(self, selected_data):
        self.selected_data = selected_data
    
    def __len__(self):
        return len(self.selected_data)
    
    def __getitem__(self, idx):
        return self.selected_data[idx]
    


class Trainer:
    def __init__(self, params, log_dir):
        """
        Initialize the Trainer.

        Parameters:
        - params (dict): Dictionary containing model, dataset, batch_size, learning_rate, and epochs.
        - log_dir (str): Directory where the TensorBoard logs will be stored.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
        self.model = params['model']
        self.train_dataset = params['train_dataset']
        self.test_dataset = params['test_dataset']  # Add a test dataset to params
        self.batch_size = params.get('batch_size', 32)
        self.learning_rate = params.get('learning_rate', 0.001)
        self.epochs = params.get('epochs', 10)

        # Create a DataLoader from the dataset
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        # Define the loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.writer = SummaryWriter(log_dir=log_dir)  

    def train(self, subset_indices=None):
        """
        Train the model for a specified number of epochs.
        """
        
        train_loss_history = []
        self.model.to(self.device)

        self.model.train() 
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            
            if subset_indices is None:
                train_dataloader = self.train_dataloader
            else:
                subset = Subset(self.train_dataset, subset_indices)
                train_dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)

            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                # Move inputs and labels to the device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Log loss value to TensorBoard for each batch
                global_step = epoch * len(train_dataloader) + batch_idx
                self.writer.add_scalar('Training Loss (Batch)', loss.item(), global_step)
                train_loss_history.append(loss.item())

                running_loss += loss.item()

                print(f"Epoch {epoch + 1}/{self.epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

            # Calculate and log the average loss for this epoch
            avg_loss = running_loss / len(train_dataloader)
            self.writer.add_scalar('Average Training Loss (Epoch)', avg_loss, epoch)

            print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {avg_loss:.4f}")

        self.writer.close()

        self.model.to("cpu")
        # Clear the GPU memory
        torch.cuda.empty_cache()

        return train_loss_history

    def eval(self):
        """
        Evaluate the model on the test set and calculate the mean and variance of the loss.

        Returns:
        - mean_loss (float): Mean of the test losses.
        - var_loss (float): Variance of the test losses.
        """

        self.model.to(self.device)
        test_loss_history = []

        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                # Move inputs and labels to the device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss_history.append(loss.item())

        return test_loss_history
        

