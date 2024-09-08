# deep-learning-challenge


Write a Report on the Neural Network Model


For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.

The overview of this neural network model is to predict and gain the accurate figures of the charities that are successful versus unsuccessful charities

Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

What variable(s) are the target(s) for your model?
IS SUCCESSFUL column

What variable(s) are the features for your model?

All the columns in the dataset except my target variable (IS_SUCCESSFUL column).

What variable(s) should be removed from the input data because they are neither targets nor features?

The EIN and NAME columns; these are just identifiers and not useful for modeling.

Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?

Neurons:110
Layers: 3
Activation: 2 - relu and sigmoid

Were you able to achieve the target model performance?
Yes at accuracy 73%

What steps did you take in your attempts to increase model performance?

Adjusted the neural network architecture by tuning hyperparameters, increasing the amount of training data, 

using different optimization algorithms, and explored different activation functions.

Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.