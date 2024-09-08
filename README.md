# deep-learning-challenge Analysis


For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.

The overview of this neural network model is to predict and gain the accurate figures of the charities that will be successful versus the unsuccessful charities. 

Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

What variable(s) are the target(s) for your model?

'IS_SUCCESSFUL' column

What variable(s) are the features for your model?

The features include all the columns in the dataset except my target variable (Target variable: 'IS_SUCCESSFUL' column).

What variable(s) should be removed from the input data because they are neither targets nor features?

To achieve more than 75% accuracy, the EIN column should be removed; this is just identifier and not useful for modeling.

Compiling, Training, and Evaluating the Model

Starter_code model Results:

# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train[0])
hidden_nodes_layer1 = 80
hidden_nodes_layer2 = 30

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation='relu'))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation='relu'))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Check the structure of the model
nn.summary()

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense (Dense)                        │ (None, 80)                  │           3,520 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 30)                  │           2,430 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 1)                   │              31 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 5,981 (23.36 KB)
 Trainable params: 5,981 (23.36 KB)
 Non-trainable params: 0 (0.00 B)

# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

268/268 - 0s - 2ms/step - accuracy: 0.7284 - loss: 0.5622
Loss: 0.5622493028640747, Accuracy: 0.728396475315094



AlphabetSoupCharity_Optimization:

number_input_features = len( X_train[0])

hidden_nodes_layer1 = 7

hidden_nodes_layer2 = 14

hidden_nodes_layer3 = 21

nn = tf.keras.models.Sequential()

# First hidden layer

nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation='relu'))

# Second hidden layer

nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation='relu'))

# Output layer

nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Check the structure of the model

nn.summary()

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense_3 (Dense)                      │ (None, 7)                   │           3,171 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_4 (Dense)                      │ (None, 14)                  │             112 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_5 (Dense)                      │ (None, 1)                   │              15 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 3,298 (12.88 KB)
 Trainable params: 3,298 (12.88 KB)
 Non-trainable params: 0 (0.00 B)

model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

268/268 - 1s - 3ms/step - accuracy: 0.7911 - loss: 0.4648
Loss: 0.4647948145866394, Accuracy: 0.7911370396614075

How many neurons, layers, and activation functions did you select for your neural network model, and why?

Neurons:21
Layers: 3
Activation: 2 - relu and sigmoid

Were you able to achieve the target model performance?
Yes at accuracy 79%

What steps did you take in your attempts to increase model performance?

Adjusted the neural network architecture by tuning hyperparameters, increasing the amount of training data, 

using different optimization algorithms, and explored different activation functions.

Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
In order to process the data the "EIN" and "NAME" column were dropped at first in the first model, then added back the name column in the AlphabetSoupCharity_Optimization model and used 'classification' for binning purposes in order to gain an accuracy of 79%. The target variable('IS_SUCCESSFUL') was model as 1 for YES and 0 for NO. Categorical variables were encoded by get_dummies() after binning was successful.

AlphabetSoupCharity_Optimization Results;
268/268 - 1s - 3ms/step - accuracy: 0.7911 - loss: 0.4648
Loss: 0.4647948145866394, Accuracy: 0.7911370396614075