from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

"""
Neural Network 
"""
def predict_using_neural_net(df):

        # Select features and target variable
        X = df[['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']]
        y = df['dataset_numeric']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19104)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Reshape to (batch_size, 1)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # Create an instance of the model
        model = NeuralNetwork()

        # Define the loss function and optimizer - Most popularly used
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        optimizer = optim.AdamW(model.parameters(), lr=0.001)  # AdamW optimizer with weight decay

        """ 
        TensorDataset: 
        
        This class is used to wrap tensors representing the input features and target labels into a single dataset object. 
        Each sample in the dataset corresponds to a pair of input features and target labels.
        
        DataLoader: 

        This class is used to create an iterable over the dataset, enabling you to iterate through batches of data during training. 
        It allows you to specify parameters such as batch size and whether to shuffle the data between epochs.

        """
        
        # Convert data to DataLoader
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # Training the model
        epochs = 50
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        # Evaluate the model
        with torch.no_grad():
            model.eval()
            outputs = model(X_test_tensor)
            predictions = (outputs >= 0.5).float()  # Thresholding at 0.5
            
            # Convert PyTorch tensors to numpy arrays with float32 data type
            predictions_np = predictions.numpy().astype('float32')
            y_test_np = y_test_tensor.numpy().astype('float32')

        return y_test_np,predictions_np
            
# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(4, 64)   # Input size: 4, Output size: 64
        self.fc2 = nn.Linear(64, 32)  # Input size: 64, Output size: 32
        self.fc3 = nn.Linear(32, 1)   # Input size: 32, Output size: 1
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x

"""
LSTM With Attention
"""

def predict_using_lstm(df):
    
    # Select features and target variable
    X = df[['d_content_average', 'd_expression_average', 'oi_content_average', 'oi_expression_average']]
    y = df['dataset_numeric']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19104)

    y_train = torch.stack(y_train)
    y_test = torch.stack(y_test)

    model = LSTMWithAttention()

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    num_epochs = 100
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs,weights = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        
    # Evaluation phase
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        test_outputs,attn_weights = model(X_test)

        # Convert model outputs to binary predictions
        preds = torch.sigmoid(test_outputs.squeeze()) >= 0.5
        
        # Convert tensors to NumPy arrays for sklearn metrics
        predictions_np = preds.numpy().astype('float32')
        y_test_np = y_test.numpy().astype('float32')

        return y_test_np,predictions_np
    
class LSTMWithAttention(nn.Module):
    def __init__(self, df,input_dim=4, hidden_dim=64, output_dim=1, num_layers=1):
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.df = df

    def forward(self, x):

        """ 
        The input sequence x is passed through the LSTM layer. lstm_out contains the LSTM's output for each time step.
        """
        lstm_out, _ = self.lstm(x) #the length of x will determine the number of timestamps. Since our input data is padded, this will be the length of the longest sequence
        # print(x.shape)

        """ 
        The LSTM output is then passed through the attention layer. 
        This layer assigns a weight to each time step of the LSTM output.
        The softmax function ensures that these weights sum up to 1, making them a valid probability distribution.
        """
        attn_weights = F.softmax(self.attention(lstm_out), dim=1) 

        """ 
        The attention weights are then used to compute a weighted sum of the LSTM outputs, which is a way to focus on the most relevant parts of the input sequence. 
        The function torch.bmm performs a batch matrix-matrix product of the attention weights and LSTM outputs.
        """
        attn_applied = torch.bmm(attn_weights.transpose(1, 2), lstm_out)

        """ 
        The attention-weighted sum is passed through the final linear layer to produce the model's output.
        """
        output = self.fc(attn_applied.squeeze(1))

        return output, attn_weights
    
""" 
Logistic Regression
"""

def predict_using_logistic_regression(df):

    # Split features and target
    X = df[['d_content_average', 'd_expression_average','oi_content_average','oi_expression_average']]
    y = df['dataset_numeric']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19104)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test,y_pred