# Required Python Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from scipy.io import arff

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    rfc = RandomForestClassifier(n_estimators=10,verbose=0)
    scores = cross_validate(rfc, features, target, cv=5, return_train_score=False)
    print ("Score notes :: ", scores)
    #                scoring=('r2', 'neg_mean_squared_error'))
    preScores = cross_val_score(rfc, features, target, cv=5)
    print ("Pre-Score notes :: ", preScores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (preScores.mean(), preScores.std() * 2))

    rfc.fit(features, target)
    return rfc

def main():
    """
    Main function
    :return:
    """

    dPath = "D:\\TCC\\WEKA\\"
    
    dFileTrain = "QAMPSKKKKK.arff"
    data = arff.loadarff(dPath+dFileTrain)
    dfTrain = pd.DataFrame(data[0])

    dFileTest = "Classifier.arff"
    data = arff.loadarff(dPath+dFileTest)
    dfTest = pd.DataFrame(data[0])
    
    #print(df.describe())

    #train_x, test_x, train_y, test_y = train_test_split(df.drop('Class', 1), df["Class"], train_size=0.9)
    train_x = dfTrain.iloc[:,0:-1]
    train_y = dfTrain.iloc[:,-1]
    test_x = dfTest.iloc[:,0:-1]
    test_y = dfTest.iloc[:,-1]

    train_y=train_y.astype('|S')
    test_y=test_y.astype('|S')

    print ("Train_x Shape :: ", train_x.shape)
    print ("Train_y Shape :: ", train_y.shape)
    print ("Test_x Shape :: ", test_x.shape)
    print ("Test_y Shape :: ", test_y.shape)

    trained_model = random_forest_classifier(train_x, train_y)

    print ("Trained model :: ", trained_model)
    predictions = trained_model.predict(test_x)

    for i in range(0, len(predictions)):
        if((list(test_y)[i].decode('UTF-8'))!=(predictions[i].decode('UTF-8'))):
            print ("Error: Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i].decode('UTF-8'), predictions[i].decode('UTF-8')))

    print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print ("Confusion matrix\n", confusion_matrix(test_y, predictions))
    
    
    multiclass = confusion_matrix(test_y, predictions)
    class_names = ['16QAM', '32QAM', '64QAM', 'BPSK', 'QPSK']

    fig, ax = plot_confusion_matrix(conf_mat=multiclass, colorbar=True, show_absolute=False, show_normed=True, class_names=class_names)
    ax.margins(2,2)
    plt.show()

if __name__ == "__main__":
    main()