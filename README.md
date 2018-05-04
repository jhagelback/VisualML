# VisualML 3.10
VisualML is a machine learning library written in Java with a GUI to visualize the decision boundaries for a classifier and how they change during training. 
The visualization is useful when teaching machine learning where students can see how different types of classifiers learn. 
The library can also be used for classification tasks, both as stand-alone application and API, on all datasets in CSV format.
VisualML is fast compared to other Java implementations of neural networks, and with comparable results.

The following classifiers are available in the library:
- k-Nearest Neighbor
- Softmax Regression linear classifier
- Neural Network
- Deep Neural Network (2 or more hidden layers)
- RBF (Radial-Basis Function) Kernel classifier
- CART (Classification And Regression Tree)
- Random Forest

## Use as stand-alone application
To run the GUI, run the VisualML.jar file without any parameters

To run a classification task, run the VisualML.jar file with the following parameters:
```
java -jar VisualML.jar -exp [id] train|test|cv
```
where [id] is the identifier for an experiment in the experiments.xml file. The next parameter is evaluation options (evaluate on training and/or test datasets and/or cross-validation).
Example:
```
java -jar VisualML.jar -exp nn_iris_test train|test
```
This trains a Neural Network classifier on the iris_train.csv dataset and evaluates accuracy on both the training and test datasets.

Experiments can also be run in the right panel in the GUI window.

A range of common datasets are available in the data folder. The MNIST hand-written characters dataset in CSV format is available in the data_mnist folder. You need to unzip the files before using them.

## Use as API
To use the library from other Java code you first need to add a new experiment for your classification task in the experiments.xml file.
Example:
```
<Experiment id="nn_iris_test">
    <Classifier>NN</Classifier>
    <TrainingData>data/iris_training.csv</TrainingData>
    <TestData>data/iris_test.csv</TestData>
    <Iterations>500</Iterations>
    <LearningRate>1.0</LearningRate>
    <UseRegularization>false</UseRegularization>
    <HiddenLayers>2</HiddenLayers>
    <Normalization>-1:1</Normalization>
</Experiment>
```
After that you build the classifier using the ClassifierFactory class:
```
Logger out = Logger.getConsoleLogger();
Classifier c = ClassifierFactory.build("nn_iris_test", out);
```
Now you can train and evaluate the accuracy on the dataset:
```
Logger out = Logger.getConsoleLogger();
c.train(out);
Metrics m = c.evaluate(true, true, out);
```
The first two parameters in the evaluate method is if the classifier shall be evaluated on the training dataset and test dataset.
The Metrics object contains various performance metrics:
```
m.getAccuracy(); //Returns the accuracy
m.getAvgPrecision(); //Returns the average precision
m.getAvgRecall(); //Returns the average recall
m.getAvgFscore(); //Returns the average F-score
m.format_conf_matrix(out); //Outputs the Confusion Matrix to an output logger
```

You can classify a new instance with:
```
String pred_label = c.classify(Instance);
```
## Dimensionality Reduction
VisualML supports dimensionality reduction using Principal-Component Analysis (PCA) and Single-Value Decomposition (SVD).

To reduce the dimensionality of a dataset, run the VisualML.jar file with the following parameters:
```
java -jar VisualML.jar -dr PCA|SVD [filename] [columns]
```
where [filename] is the path to the dataset file and [columns] is the number of columns to keep (dimensionality) if PCA 
is used. The dataset will be saved in a new dataset file in the same folder as the original dataset file.

Example:
```
java -jar VisualML.jar -dr PCA data/iris.csv 2
```
This reduces the number of attributes in the iris dataset to 2 using PCA, and saves the new dataset as data/iris_pca.csv.

```
java -jar VisualML.jar -dr SVD data/iris.csv
```
This reduces the number of attributes in the iris dataset using SVD, and saves the new dataset as data/iris_svd.csv.
