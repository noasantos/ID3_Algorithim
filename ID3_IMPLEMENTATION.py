import numpy as np
import pandas as pd

#Main-Class
class Node:
    def __init__(self, attributeValue, count):
        self.attributeValue = attributeValue
        self.count = count

#Sub-Class: Attribute      
class Attribute(Node):
    def __init__(self, attribute=None, isDiscreteType=None, majorTargetValue=None, attributeValue=None, count=None):
        super().__init__(attributeValue, count)
        self.attribute = attribute 
        self.isDiscreteType = isDiscreteType
        self.majorTargetValue = majorTargetValue
        self.children = []
    
    def addLeafNode(self, attributeValue, targetValue, count):
        child = Leaf(attributeValue, targetValue, count)
        self.children.append(child)
        return child
    
    def addAttributeNode(self, attribute, isDiscreteType, majorTargetValue, attributeValue, count):
        child = Attribute(attribute, isDiscreteType, majorTargetValue, attributeValue, count)
        self.children.append(child)
        return child 

#Sub-Class: Leaf
class Leaf(Node):
    def __init__(self, targetValue, attributeValue, count=None):
        super().__init__(attributeValue, count)
        self.targetValue = targetValue

#        
def buildTree(dataframe: pd.DataFrame, parentNode=None, uniqueValues=None):
    target, attributes = dataframe.iloc[:, -1], dataframe.iloc[:, :-1]
    bestAttribute, bestSplit = getMostInformativeFeature(attributes, target)
    isDiscreteType = True if bestSplit is None else False
    
    if attributes.empty:
        return
    elif parentNode is None: #rootNode need to be created
        parentNode = Attribute(bestAttribute, isDiscreteType, target.value_counts().idxmax(), attributeValue=None, count=len(attributes))
        uniqueValues = {col: dataframe[col].unique().tolist() for col in dataframe}
    else:
        parentNode.attribute = bestAttribute
        parentNode.isDiscreteType = isDiscreteType
    
    if isDiscreteType: #Column Type is Discrete
        currentValues = attributes[bestAttribute].unique()
        allAttributeValues = uniqueValues[bestAttribute]
        discreteChildren(dataframe, parentNode, currentValues, uniqueValues)
        leafForValueNotInCurrValues(parentNode, currentValues, allAttributeValues)
    else: #Column Type is Continuous
        continuousChildren(dataframe, parentNode, bestSplit, uniqueValues)
        
    return parentNode

#
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

#
def discreteChildren(dataframe, parentNode, currentValues, uniqueValues):
    for attributeValue in currentValues:
            childDataFrame = (dataframe[dataframe[parentNode.attribute] == attributeValue]).drop(columns=parentNode.attribute)
            
            createNewNode(parentNode, childDataFrame, attributeValue, uniqueValues)
    return

#
def leafForValueNotInCurrValues(parentNode, currentValues, allAttributeValues):
    for attributeValue in allAttributeValues:
        if attributeValue not in currentValues:
            parentNode.addLeafNode(parentNode.majorTargetValue, attributeValue, 0)
    return

#
def createNewNode(parentNode, childDataFrame, attributeValue, uniqueValues):
    counts = childDataFrame.iloc[:, -1].value_counts()
    count = len(childDataFrame)
    if isNodePure(counts) or not isAttributesAvailable(childDataFrame):
        targetValue = counts.idxmax()
        parentNode.addLeafNode(targetValue, attributeValue, count)
    else:
        attributeNode = parentNode.addAttributeNode(attribute=None, isDiscreteType=None, count=count, majorTargetValue=counts.idxmax(), attributeValue=attributeValue)   
        buildTree(childDataFrame, attributeNode, uniqueValues)
    return

def isAttributesAvailable(childDataFrame):
    return childDataFrame.shape[1] > 1
 
def isNodePure(counts):
    return len(counts) == 1

#Universal Entropy Calculation
def entropy(target: pd.DataFrame):
    counts = target.value_counts()
    probs = counts/len(target)
    return np.sum(-probs * np.log2(probs))


#Continous Split Function for Calculation Entropy
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

#Discrete Split Function for Calculation Entropy
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