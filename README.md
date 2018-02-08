# VisualML 3.0
Visual ML is a machine learning library written in Java with a GUI to visualize the decision boundaries for a classifier and how they change during training. 
The visualization is useful when teaching machine learning where students can see how different types of classifiers learns. 
The library can also be used for classification tasks, both as stand-alone application and API, on all datasets in CSV format.
VisualML is fast compared to other Java implementations of neural networks, and with comparable results.

The following classifiers are available in the library:
- k-Nearest Neighbor
- Softmax Linear classifier
- Neural Network
- Deep Neural Network (2 or more hidden layers)

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
Classifier c = ClassifierFactory.build("nn_iris_test");
```
Now you can train and evaluate the accuracy on the dataset:
```
Logger out = Logger.getConsoleLogger();
c.train(out);
c.evaluate(true, true, out);
```
The first two parameters in the evaluate method is if the classifier shall be evaluated on the training dataset and test dataset.

You can classify a new instance with:
```
String pred_label = c.classify(Instance);
```
