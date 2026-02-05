# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY

Neural Network Regression is a supervised learning technique used to predict continuous output values. Unlike linear regression, neural networks can learn non-linear relationships between input features and the target variable.

A neural network consists of an input layer, one or more hidden layers, and an output layer. Each neuron multiplies inputs by weights, adds a bias, and applies an activation function. Hidden layers usually use ReLU, while the output layer uses a linear activation for regression.

The model is trained using backpropagation, where the error between predicted and actual values is calculated using a loss function such as Mean Squared Error (MSE). The weights are updated using an optimizer like Gradient Descent or Adam to minimize the error.

Neural network regression is widely used in price prediction, forecasting, and data analysis due to its ability to handle complex datasets.

## Neural Network Model

<img width="566" height="392" alt="image" src="https://github.com/user-attachments/assets/1aaae0f7-7a4a-4492-a9a9-6dcd9dcbf258" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: RESHMITHAA B

### Register Number: 212224220080

```python
# Name: RESHMITHAA B
# Register Number: 212224220080
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)



# Name: RESHMITHAA B
# Register Number: 212224220080
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      outputs = ai_brain(X_train)
      loss = criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```

### Dataset Information

<img width="138" height="274" alt="image" src="https://github.com/user-attachments/assets/5078cfb1-070f-4ad4-a396-bd459392c259" />

### OUTPUT

<img width="410" height="186" alt="image" src="https://github.com/user-attachments/assets/455b34e9-5c77-4b04-8d5b-012d6bc9a195" />

<img width="254" height="35" alt="image" src="https://github.com/user-attachments/assets/722c8831-906e-4a78-acd4-acb99a1cbd9d" />



### Training Loss Vs Iteration Plot

<img width="594" height="409" alt="image" src="https://github.com/user-attachments/assets/10bf7274-55e2-46cd-bf91-b4c3e0f21f46" />


### New Sample Data Prediction

<img width="341" height="32" alt="image" src="https://github.com/user-attachments/assets/b51da11b-18ae-4a5d-8c9d-43f9855a86d8" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
