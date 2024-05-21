from ID3_IMPLEMENTATION import *
from ID3_METRICS import *
from ID3_VIZUALIZATION import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def runID3_TRAINTEST(dataframe):
    trainData, testData = train_test_split(dataframe, test_size=0.2, random_state=42)
    rootNode = buildTree(trainData)
    testData = addPredictedColumn(rootNode, testData)
    accuracy = calculateAccuracy(testData)
    return rootNode, accuracy

def runID3_KFold(dataframe, k=4):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    
    for train_index, test_index in kf.split(dataframe):
        trainData, testData = dataframe.iloc[train_index], dataframe.iloc[test_index]
        rootNode = buildTree(trainData)
        testData = addPredictedColumn(rootNode, testData)
        accuracy = calculateAccuracy(testData)
        accuracies.append(accuracy)
    
    mean_accuracy = sum(accuracies) / len(accuracies)
    return mean_accuracy

def numFolds(len):
    numTest_df = int(len*0.10) if len > 30 else int(len*0.25)
    return int(len/numTest_df)

def preProcess(datasource):
    dataframe = pd.read_csv(datasource, keep_default_na=False)
    if 'ID' in dataframe.columns:
        dataframe.set_index('ID', inplace=True)   
    return dataframe

def runID3_ALLDATA(dataframe):
    rootNode = buildTree(dataframe)
    return rootNode

def run(datasource, type):
    dataframe = preProcess(datasource)
    rootNode = None
    
    if type == "KFOLD":
        k = numFolds(len(dataframe))
        mean_accuracy = runID3_KFold(dataframe, k)
        print(f"Finished! K-Fold Cross-Validation Accuracy: {mean_accuracy}")
    elif type == "TRAINTEST":
        rootNode, accuracy = runID3_TRAINTEST(dataframe)
        print(f"Finished! One-Test Accuracy: {accuracy}")
    elif type == "ALLDATA":
        print("Finished!")
        rootNode = runID3_ALLDATA(dataframe)
    
    if rootNode is not None:
        return rootNode
    