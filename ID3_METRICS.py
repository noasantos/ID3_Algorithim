from ID3_IMPLEMENTATION import Leaf
import pandas as pd
from sklearn.metrics import accuracy_score

def predictClass(node, rowData):
    if isinstance(node, Leaf):
        return node.targetValue
    
    for child in node.children:
        if node.isDiscreteType:
            if child.attributeValue == rowData[node.attribute]:
                return predictClass(child, rowData)
        else:
            if child.attributeValue.startswith("<="):
                if rowData[node.attribute] <= float(child.attributeValue.lstrip("<=")):
                    return predictClass(child, rowData)
            elif child.attributeValue.startswith(">"):
                if rowData[node.attribute] > float(child.attributeValue.lstrip(">")):
                    return predictClass(child, rowData)


def addPredictedColumn(rootNode, testData: pd.DataFrame):
    testData['Predicted Class'] = testData.apply(lambda row: predictClass(rootNode, row), axis=1)
    return testData     

def calculateAccuracy(testData):
    trueValue, predictedValue = testData.iloc[:, -2], testData.iloc[:, -1]
    return accuracy_score(trueValue, predictedValue)  