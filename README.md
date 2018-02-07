# VisualML
Visual ML is a machine learning library written in Java with a GUI to visualize the decision boundaries for a classifier and how they change during training. 
The visualization is useful when teaching machine learning where students can see how different types of classifiers learns. 
The library can also be used for classification tasks, both as stand-alone application and API, on all datasets in CSV format.

The following classifiers are available in the library:
- k-Nearest Neighbor
- Softmax Linear classifier
- Neural Network
- Deep Neural Network (2 or more hidden layers)

## Usage
To run the GUI, run the VisualML.jar file without any parameters

To run a classification task, run the VisualML.jar file with the following parameters:
```
java -jar VisualML.jar -exp [id]
```
where [id] is the identifier for an experiment in the experiments.xml file.
Example:
```
java -jar VisualML.jar -exp iris_test
```
