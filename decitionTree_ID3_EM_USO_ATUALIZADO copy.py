import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from graphviz import Digraph

class Tree:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe
        self.uniqueValues = {col: dataframe[col].unique().tolist() for col in dataframe}

class Node:
    def __init__(self, edgeValue=None, count=None):
        self.edgeValue = edgeValue
        self.count = count
        self.children = []
    
    def addLeafNode(self, attributeValue, targetValue, count):
        child = Leaf(attributeValue, targetValue, count)
        self.children.append(child)
        return child
        
    def addAttributeNode(self, attribute, isDiscreteType, count, majorTargetValue, edgeLabel):
        child = Attribute(attribute, isDiscreteType, count, majorTargetValue, edgeLabel)
        self.children.append(child)
        return child
    
class Attribute(Node):
    def __init__(self, attribute=None, isDiscreteType=None, count=None, majorTargetValue=None, attributeValue=None):
        super().__init__(type)
        self.attribute = attribute 
        self.isDiscreteType = isDiscreteType
        self.count = count
        self.attributeValue = attributeValue
        
    def setAttribute(self, attribute):
        self.attribute = attribute
        
    def setIsDiscreteType(self, typeBool):
        self.isDiscreteType = typeBool        

class Leaf(Node):
    def __init__(self, attributeValue, targetValue, count=None):
        super().__init__(type)
        self.attributeValue = attributeValue #edgeLabel
        self.targetValue = targetValue
        self.count = count

class InternalNode(Node):
    def __init__(self, attributeValue, count):
        super().__init__(type)
        self.attributeValue = attributeValue 
        self.count = count


        
def buildTree(dataframe: pd.DataFrame, parentNode=None, uniqueValues=None):
    print(parentNode)
    target, attributes = dataframe.iloc[:, -1], dataframe.iloc[:, :-1]
    bestAttribute, bestSplit = getMostInformativeFeature(attributes, target)
    isDiscreteType = True if bestSplit is None else False
    
    if attributes.empty:
        return
    elif parentNode is None: #rootNode need to be created
        parentNode = Attribute(bestAttribute, isDiscreteType, len(attributes), target.value_counts().idxmax())
        uniqueValues = {col: dataframe[col].unique().tolist() for col in dataframe}
    else:
        parentNode.setAttribute(bestAttribute)
        parentNode.setIsDiscreteType(isDiscreteType)
    
    if isDiscreteType: #Column Type is Discrete
        currentValues = attributes[bestAttribute].unique()
        allAttributeValues = uniqueValues[bestAttribute]
        discreteChildren(dataframe, parentNode, currentValues, uniqueValues)
        leafForValueNotInCurrValues(parentNode, currentValues, allAttributeValues)
    else: #Column Type is Continuous
        continuousChildren(dataframe, parentNode, bestSplit, uniqueValues)
        
    return parentNode

def continuousChildren(dataframe, parentNode, bestSplit, uniqueValues):
    dfLessEqual = (dataframe.loc[dataframe[parentNode.attribute] <= bestSplit]).drop(columns=parentNode.attribute)
    dfGreater = (dataframe.loc[dataframe[parentNode.attribute] > bestSplit]).drop(columns=parentNode.attribute)
    for childDataFrame in [dfLessEqual, dfGreater]: 
        if childDataFrame.equals(dfLessEqual):
            attributeValue = "<=" + str(bestSplit)
        else:
            attributeValue = ">" + str(bestSplit)
        createNewNode(parentNode, childDataFrame, attributeValue, uniqueValues)
    return

def discreteChildren(dataframe, parentNode, currentValues, uniqueValues):
    for attributeValue in currentValues:
            childDataFrame = (dataframe[dataframe[parentNode.attribute] == attributeValue]).drop(columns=parentNode.attribute)
            
            createNewNode(parentNode, childDataFrame, attributeValue, uniqueValues)
    return

def leafForValueNotInCurrValues(parentNode, currentValues, allAttributeValues):
    for attributeValue in allAttributeValues:
        if attributeValue not in currentValues:
            parentNode.addLeafNode(attributeValue, parentNode.majorTargetValue, 0)
    return

def createNewNode(parentNode, childDataFrame, attributeValue, uniqueValues):
    counts = childDataFrame.iloc[:, -1].value_counts()
    count = len(childDataFrame)
    if isNodePure(counts) or not isAttributesAvailable(childDataFrame):
        targetValue = counts.idxmax()
        parentNode.addLeafNode(attributeValue, targetValue, count)
    else:
        attributeNode = parentNode.addAttributeNode(attribute=None, isDiscreteType=None, count=count, majorTargetValue=counts.idxmax(), edgeLabel=attributeValue)   
        buildTree(childDataFrame, attributeNode, uniqueValues)
    return

def isAttributesAvailable(childDataFrame):
    return childDataFrame.shape[1] > 1
 
def isNodePure(counts):
    return len(counts) == 1



#Standard Entropy
def entropy(target: pd.DataFrame):
    counts = target.value_counts()
    probs = counts/len(target)
    return np.sum(-probs * np.log2(probs))


#Continous Entropy
def continousEntropy(feature, target):
    dfTemp = pd.concat([feature, target], axis=1)
    entropyResults = {}
    inverseEntropyResults = {}
    uniqueValuesList = sorted(feature.unique().tolist())
    
    for value in uniqueValuesList:
        totalWeightedEntropy = splitEntropy(dfTemp, value)
        
        entropyResults[value] = totalWeightedEntropy
        inverseEntropyResults[totalWeightedEntropy] = value
    
    smallestEntropy = min(entropyResults.values())
    splitPoint = inverseEntropyResults[smallestEntropy]
    return smallestEntropy, splitPoint
            
def splitEntropy(df: pd.DataFrame, splitPoint):
    feature, target = df.columns
    dfLessEqual = df.loc[df[feature] <= splitPoint]
    dfGreater = df.loc[df[feature] > splitPoint]
    
    lessEqualEntropy = entropy(dfLessEqual[target])
    greaterEntropy = entropy(dfGreater[target])
    n = len(df)
    totalWeightedEntropy = lessEqualEntropy * len(dfLessEqual)/n + greaterEntropy * len(dfGreater)/n
    return totalWeightedEntropy

#Discrete Entropy
def  discreteEntropy(feature, target):
    dfTemp = pd.concat([feature, target], axis=1)
    uniqueValues = feature.unique()
    totalWeightedEntropy = 0
    
    for value in uniqueValues:
        subSet = dfTemp[dfTemp[feature.name] == value]
        
        totalWeightedEntropy += len(subSet) / len(dfTemp) * entropy(subSet[target.name])
    
    return totalWeightedEntropy

#Information Gain Function
def informationGain(feature, target, entrophyBefore):
    continuousSplitPoint = None
    if pd.api.types.is_numeric_dtype(feature):
        weightedEntrophyAfter, continuousSplitPoint= continousEntropy(feature, target)
    else: 
        weightedEntrophyAfter = discreteEntropy(feature, target)
    infoGain = entrophyBefore - weightedEntrophyAfter
    return infoGain, continuousSplitPoint

#Find the best column between all Function
def getMostInformativeFeature(attributes, target):
    entrophyBefore = entropy(target)
    maxInfoGain = -1
    bestInfoAttribute = None
    
    for _, attributeCol in enumerate(attributes):
        colInfoGain, splitPoint = informationGain(attributes[attributeCol], target, entrophyBefore)
        if colInfoGain > maxInfoGain:
            maxInfoGain = colInfoGain
            bestInfoAttribute = attributeCol
            bestSplit = splitPoint
    return bestInfoAttribute, bestSplit

#Print Tree Function
def printTree(node, depth=0, attributeChild=True):
    if node is None:
        return

    indent = " " * depth * 5
    if isinstance(node, Attribute): #se for o nó pai da feature
        print(indent + f"Attribute: {node.attribute} | Value: {node.attributeValue}")
    elif isinstance(node, Leaf):
        print(indent + f"{node.attributeValue}: {node.targetValue} ({node.count}*)") #se for um nó filho que é uma folha
    
    for child in node.children:
        printTree(child, depth+1)

#Function to predict the target value for test dataset
def predictClass(node, rowData):
    if node.classLabel is not None:
        return node.classLabel
    elif node.value is not None:
        node = node.children[0]
    
    for child in node.children:
        if child.value.startswith("<="):
            if rowData[node.feature] <= float(child.value.lstrip("<=")):
                return predictClass(child, rowData)
        elif child.value.startswith(">"):
            if rowData[node.feature] > float(child.value.lstrip(">")):
                return predictClass(child, rowData)
        elif child.value == rowData[node.feature]:
            return predictClass(child, rowData)

def addPredictedColumn(rootNode, testData: pd.DataFrame):
    predicted = []
    
    for _, row in testData.iterrows():
        predictedClass = predictClass(rootNode, row)
        predicted.append(predictedClass)
    testData['Predicted Class'] = predicted
    return testData     

def calculateAccuracy(testData):
    return accuracy_score(testData.iloc[:, -2], testData.iloc[:, -1])  

#Run de ID3 algorithim
def runID3(trainData, testData):
    rootNode = buildTree(trainData)
    printTree(rootNode, )
    testData = addPredictedColumn(rootNode, testData)
    print(testData)
    return calculateAccuracy(testData)

def run(datasource):
    df = preProcess(pd.read_csv(datasource))
    rootNode = buildTree(df)
    printTree(rootNode, attributeChild=False)
    
    dot = exportTree(rootNode)
    dot.render("arvore_decisao_weather", format="png")
    dot.view()

def exportTree(node, dot=None, parent_name=None, edge_label=""):
    if dot is None:
        dot = Digraph()
    
    if isinstance(node, Attribute):
        node_label = f"{node.attribute}"
    elif isinstance(node, Leaf):
        node_label = f"{node.targetValue}\n{node.count}"
    
    curr_node_name = f"{id(node)}" 
    dot.node(curr_node_name, label=node_label, shape="ellipse" if isinstance(node, Leaf) else "box")
    
    if parent_name is not None:
        dot.edge(parent_name, curr_node_name, label=edge_label)
    
    for child in node.children:
        edge_label = child.attributeValue
        exportTree(child, dot, curr_node_name, edge_label)
                   
    return dot


def prepare(dataSource):
    df = preProcess(pd.read_csv(dataSource))
    trainData, testData = train_test_split(df, test_size=0.2, random_state=42)
    
    accuracy = runID3(trainData, testData)
    print(accuracy)
    
def preProcess(dataframe: pd.DataFrame):
    if 'ID' in dataframe.columns:
        dataframe.set_index('ID', inplace=True)   
    return dataframe

run('connect4.csv')
#prepare('restaurant.csv')

        
    
    
    
    
    
    
    
    
    
    