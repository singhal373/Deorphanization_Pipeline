import pandas as pd
import numpy as np
import sklearn
import sys
from imblearn.pipeline import Pipeline as sample_pipeline
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import smote_variants as sv
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef 
from sklearn.manifold import TSNE
from sklearn.model_selection import LeaveOneOut 
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA 
import smote_variants as sv
from sklearn.linear_model import LogisticRegression
import numpy as np
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
from rdkit import rdBase
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Fingerprints import FingerprintMols
import matplotlib.backends.backend_pdf

ORName=input("Enter OR Name")
outputfile = input("Enter File  Name to Save Output: ") 
f = open(outputfile,'w'); 
sys.stdout = f
figcount=0
Figureset=[]
#Important Info On using Smote with pipeline #https://stackoverflow.com/questions/50245684/using-smote-with-gridsearchcv-in-scikit-learn

class Data_split: # stratified split of the dataset given test size as input
    def splitdataset(self,fulldata,fulldatalabel,k):
        X_train, X_test, y_train, y_test = train_test_split(fulldata,fulldatalabel,test_size=k,random_state=100,stratify=fulldatalabel)
        return X_train,X_test,y_train,y_test

class Data_Read:
    def Read_Data(self,filepath): # Input data should have one 1 column "Activation_Status" 1 column named "Ligand"
        data=pd.read_csv(filepath)
        data_labels=data["Activation Status"]
        Ligand_names=data["Ligand"]
        data=data.drop("Activation Status",axis=1)
        data=data.drop("Ligand",axis=1)
        try:
            data=data.drop["Smiles"]
        except:
            print("DO Smile Column")
        data=self.pruneColumns(data)
        return data,data_labels

    def pruneColumns(self,data):
        data=data.replace(r'\s+', np.nan, regex=True)
        data[data==np.inf]=np.nan
        data=data.replace(r'^\s*$', np.nan, regex=True)
        data.isna().sum().to_csv("NAN_values1.csv",header=False)
        NAN_data=pd.read_csv("NAN_values1.csv",header=None)
        dropped=[]
        for i in range(len(NAN_data)):
            if NAN_data.iloc[i][1] >=75:
                dropped.append(NAN_data.iloc[i][0])
        data=data.drop(dropped,axis=1)
        return data  


class Preprocess_Data:

    def VarianceRemoval(self,data_normal,test_normal,thresh):
      selector = VarianceThreshold(thresh)
      selector.fit(data_normal)
      data_var_free=data_normal[data_normal.columns[selector.get_support(indices=True)]]
      test_var_free=test_normal[test_normal.columns[selector.get_support(indices=True)]]
      return data_var_free,test_var_free

    def correlation_check(self,traindata,testdata,thresh): # drop columns above certain threshold
        corr_matrix = traindata.corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] >thresh)]
        trainset=traindata.drop(traindata[to_drop], axis=1)
        testset=testdata.drop(testdata[to_drop],axis=1)
        return trainset,testset

    def handlemissingvalue(self,X_train,X_test):
        X_train=X_train.replace([np.inf, -np.inf,""," "], np.nan)
        X_train=X_train.replace([""," "],np.nan)
        X_test=X_test.replace([np.inf, -np.inf,""," "], np.nan)
        X_test=X_test.replace([""," "],np.nan)
        X_train.fillna(X_train.mean(), inplace=True)
        X_test.fillna(X_train.mean(),inplace=True)
        return X_train,X_test


    def normalize(self,traindata,testdata):  
        try:
            x = traindata.values #returns a numpy array
        except:
            x=traindata
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(x)
        x_scaled = min_max_scaler.transform(x)
        test=testdata.values                                    
        test=min_max_scaler.transform(test)
        train_normal = pd.DataFrame(x_scaled)
        test_normal=pd.DataFrame(test)
        train_normal.columns =list(traindata.columns.values)
        test_normal.columns=list(testdata.columns.values)
        return train_normal,test_normal


    def Process_data(self,traindata,testdata,corr_th,var_th):
        traindata,testdata=self.handlemissingvalue(traindata,testdata)
        traindata,testdata=self.normalize(traindata,testdata)
        traindata,testdata=self.VarianceRemoval(traindata,testdata,var_th)
        traindata,testdata=self.correlation_check(traindata,testdata,corr_th)
        return traindata,testdata

    def Process_data(self,traindata,testdata):
        traindata,testdata=self.handlemissingvalue(traindata,testdata)
        traindata,testdata=self.normalize(traindata,testdata)
        traindata,testdata=self.VarianceRemoval(traindata,testdata,0.0)
        traindata,testdata=self.correlation_check(traindata,testdata,0.95)
        return traindata,testdata

class Correlation_Filter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.X=None
        self.y=None
        self.todrop=None
    def fit(self, X,y=None):
        corr_matrix = X.corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        self.to_drop = [column for column in upper.columns if any(upper[column] >0.95)]
        return self
    def transform(self, X,y=None):
        X=X.drop(X[self.to_drop], axis=1)
        return X
    def DroppedFeatures(self):
        return self.to_drop

class pca(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.X=None
        self.y=None
        self.clf=None
    def fit(self, X,y=None):
        self.clf=PCA(0.98)
        self.clf.fit(X)
        return self
    def transform(self, X,y=None):
        X=self.clf.transform(X)
        return X


class HandleMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.X=None
        self.y=None
        self.mean=None
    def fit(self, X,y=None):
        self.mean=X.mean()
        return self
    def transform(self, X,y=None):
        X.fillna(self.mean,inplace=True)
        return X

class VarianceFilter( BaseEstimator, TransformerMixin ):
    def __init__(self):
        self.X=None
        self.y=None
        self.selector = VarianceThreshold()
    def fit(self, X,y=None): 
        self.selector.fit(X)
        return self
    def transform(self, X,y=None):
        X=X[X.columns[self.selector.get_support(indices=True)]]
        return X
    def chosenFeatures(self):
        return self.selector.get_support(indices=True)
    def chosenColumnNames(self):
        return self.selector.get_support()


class Normalize(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.X=None
        self.y=None
        self.min_max_scaler = preprocessing.MinMaxScaler()
    def fit(self, X,y=None): 
        self.min_max_scaler.fit(X)
        return self
    def transform(self, X,y=None):
        X_data=X.copy()
        X=self.min_max_scaler.transform(X)
        X =pd.DataFrame(X)
        if X_data is pd.DataFrame():
            X.columns=X_data.columns
        return X


class FeatureSelector( BaseEstimator, TransformerMixin ):
    def __init__(self):
        self.rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1) 
        self.boruta_selector = BorutaPy(self.rfc, n_estimators='auto',random_state=50)
        self.X=None
        self.cols=None
    def fit( self, X, y):
        self.cols=X.columns
        self.boruta_selector.fit(X.values,y)
        return self
    def transform( self, X,y=None): 
        X=self.boruta_selector.transform(X.values)
        return X
    def get_feature_names(self):
        cols=self.cols[self.boruta_selector.support_]
        print("IN FeatureSelector get_feature_names ",cols)
        return cols
        

class Cross_validate_GridSearch:

    def SVM_GridSearch(self):
        random.seed(50)
        Cs = [0.0001,0.001, 0.01, 0.1, 1, 10]
        gammas = [0.000001,0.0001,0.001, 0.01, 0.1, 1,10]
        kernel = ['rbf','poly','linear']
        param_grid = {'C': Cs, 'gamma' : gammas, 'kernel' : kernel}
        clf=svm.SVC(probability=True)
        grid_search = GridSearchCV(clf, param_grid, cv=5,n_jobs=-1,scoring='f1',verbose=3)
        return grid_search

    def MLP_classifier_Gridsearch(self):
        parameter_space = {'hidden_layer_sizes': [(5,5,5),(20,30,50),(50,50,50), (50,100,50), (100,),(100,100,100),(5,2)],'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05,0.001,0.01],
        'learning_rate': ['constant','adaptive']}
        mlp = MLPClassifier(max_iter=1000,random_state=50)
        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5,scoring='f1',verbose=2)
        return clf
        
    def RandomForest_GridSearch(self):
        n_estimators = [int(x) for x in np.linspace(start = 2, stop = 100, num = 10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=50, n_jobs = -1)
        return rf_random

    def printConfusionMatrix(self,testlabel,y_pred,title_name):
        global figcount
        global Figureset
        cf=confusion_matrix(testlabel,y_pred)
        df_cm = pd.DataFrame(cf, index = [0,1],columns = [0,1])
        ax= plt.figure(figcount)
        plt.title(title_name)
        sns.heatmap(df_cm, annot=True)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        figcount+=1
        Figureset.append(ax)
        plt.show()

    def printROCplot(self,testlabel,y_prob,title_name):
        global figcount
        global Figureset
        fpr, tpr, _ = roc_curve(testlabel,y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        print("AUC VALUE :",roc_auc)
        image=plt.figure(figcount)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.title(title_name)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title_name)
        plt.legend(loc="lower right")
        figcount+=1
        Figureset.append(image)
        plt.show()

    def GNBLR(self,trainset,trainlabel,testset,testlabel):
        model1=LogisticRegression(random_state=50, solver='liblinear',penalty='l2',max_iter=100)
        model2=GaussianNB()
        model=[]
        F1scores=[]
        model.append(model1)
        model.append(model2)
        model_name=["LogisticRegression","GaussianNB"]
        for i in range(len(model)):
            p2=Pipeline([("HandleMissingValues",HandleMissingValues()),("Normalize",Normalize()),("VarianceFilter",VarianceFilter()),("CorrelationFilter",Correlation_Filter()),('feature_sele',FeatureSelector()),("model",model[i])])
            print("__________________________________________________________________________________________")
            print("Model Used \n",model)
            print("RESULT OF 5 fold Cross Validation")
            scoring = {'acc': 'accuracy','prec_macro': 'precision_macro','rec_micro': 'recall_macro'}
            scores = cross_validate(p2,trainset,trainlabel,scoring=scoring,cv=5, return_train_score=True)
            print(scores)
            p2.fit(trainset,trainlabel)
            y_pred_train=p2.predict(trainset)
            y_pred_test=p2.predict(testset)
            y_pred_prob=p2.predict_proba(testset)
            print("ACCURACY ON TRAIN SET :",accuracy_score(trainlabel,y_pred_train))
            print("ACCURACY ON TEST SET",accuracy_score(testlabel,y_pred_test))
            print(classification_report(testlabel,y_pred_test))
            print("CONFUSION_MATRIX")
            F1scores.append(f1_score(testlabel, y_pred_test, average='macro'))
            self.printConfusionMatrix(testlabel,y_pred_test,model_name[i])
            self.printROCplot(testlabel,y_pred_prob,model_name[i])
            print("MCC",matthews_corrcoef(testlabel, y_pred_test))
        return F1scores
        
    def GNBLR_smote(self,trainset,trainlabel,testset,testlabel):
        global figcount
        global Figureset
        model1=LogisticRegression(random_state=50, solver='liblinear',penalty='l2',max_iter=100)
        model2=GaussianNB()
        model=[]
        F1scores=[]
        model.append(model1)
        model.append(model2)
        model_name=["LogisticRegression_withSmote","GaussianNB_withSmote"]
        for i in range(len(model)):
            p2=sample_pipeline([("HandleMissingValues",HandleMissingValues()),("sampling",SMOTE(random_state=50)),("Normalize",Normalize()),("VarianceFilter",VarianceFilter()),("CorrelationFilter",Correlation_Filter()),('feature_sele',FeatureSelector()),("model",model[i])])
            print("__________________________________________________________________________________________")
            print("Model Used \n",model)
            print("RESULT OF 5 fold Cross Validation")
            scoring = {'acc': 'accuracy','prec_macro': 'precision_macro','rec_micro': 'recall_macro'}
            scores = cross_validate(p2,trainset,trainlabel,scoring=scoring,cv=5, return_train_score=True)
            print(scores)
            p2.fit(trainset,trainlabel)
            y_pred_train=p2.predict(trainset)
            y_pred_test=p2.predict(testset)
            y_pred_prob=p2.predict_proba(testset)
            print("ACCURACY ON TRAIN SET :",accuracy_score(trainlabel,y_pred_train))
            print("ACCURACY ON TEST SET",accuracy_score(testlabel,y_pred_test))
            print(classification_report(testlabel,y_pred_test))
            print("CONFUSION_MATRIX")
            F1scores.append(f1_score(testlabel, y_pred_test, average='macro'))
            self.printConfusionMatrix(testlabel,y_pred_test,model_name[i])
            self.printROCplot(testlabel,y_pred_prob,model_name[i])
            print("MCC",matthews_corrcoef(testlabel, y_pred_test))
        return F1scores

    def EvaluateDifferentModels(self,trainset,trainlabel,testset,testlabel):
        global figcount
        global Figureset
        svm_gridsearch=self.SVM_GridSearch()
        rf_gridsearch=self.RandomForest_GridSearch()
        mlp_gridsearch=self.MLP_classifier_Gridsearch()
        grid_search_models=[svm_gridsearch,rf_gridsearch,mlp_gridsearch]
        best_models=[]
        Saving_Model=[]
        F1_scores_=[]
        Models_used=["SVM","RF","MLP","VC","LR","GNB"]
        k=-1
        for grid_search_model in grid_search_models:
            k=k+1
            pipeline  = Pipeline([("HandleMissingValues",HandleMissingValues()),("Normalize",Normalize()),("VarianceFilter",VarianceFilter()),("CorrelationFilter",Correlation_Filter()),('feature_sele',FeatureSelector()),('clf_cv',grid_search_model)])
            pipeline.fit(trainset,trainlabel)
            print(pipeline.named_steps['feature_sele'].get_feature_names())
            chosen_features=pipeline.named_steps['feature_sele'].get_feature_names()
            chosen_model=pipeline.named_steps['clf_cv'].best_estimator_
            p2=Pipeline([("HandleMissingValues",HandleMissingValues()),("Normalize",Normalize()),("VarianceFilter",VarianceFilter()),("CorrelationFilter",Correlation_Filter()),('feature_sele',FeatureSelector()),("model",pipeline.named_steps['clf_cv'].best_estimator_)])
            best_models.append(("model"+str(k),chosen_model))
            Saving_Model.append(pipeline.named_steps['clf_cv'].best_estimator_) # Appending best estimator
            y_pred_train=pipeline.predict(trainset)
            print("ACCURACY ON TRAIN SET :",accuracy_score(trainlabel,y_pred_train))
            y_pred=pipeline.predict(testset)
            y_prob=pipeline.predict_proba(testset)
            print("The important Features are :",chosen_features)
            print("Classification Report for TEST SET Using ",chosen_model)
            print(classification_report(testlabel,y_pred))
            print("ACCURACY ON TEST SET :",accuracy_score(testlabel,y_pred))
            F1_scores_.append(f1_score(testlabel,y_pred, average='macro'))
            print("CONFUSION_MATRIX")
            self.printConfusionMatrix(testlabel,y_pred,Models_used[k])
            self.printROCplot(testlabel,y_prob,Models_used[k])
            print("MCC",matthews_corrcoef(testlabel, y_pred))
            print("__________________________________________________________________________________________")
            print("RESULT OF 5 fold Cross Validation")
            scoring = {'acc': 'accuracy','prec_macro': 'precision_macro','rec_micro': 'recall_macro'}
            scores = cross_validate(p2,trainset,trainlabel,scoring=scoring,cv=5, return_train_score=True)
            print(scores)
        print("USING A VOTING ENSEMBLE WITH THE ABOVE BEST CLASSIFIERS")
        ensemble = VotingClassifier(best_models,n_jobs=-1,voting='soft')
        traindata,testdata=Preprocess_Data().normalize(trainset,testset)
        ensemble.fit(traindata[traindata.columns[chosen_features]],trainlabel)
        Saving_Model.append(ensemble) # Appending Voting model
        y_pred_ensemble=ensemble.predict(testdata[testdata.columns[chosen_features]])
        y_pred_ensemble_train=ensemble.predict(traindata[traindata.columns[chosen_features]])
        y_pred_prob_ensemble=ensemble.predict_proba(testdata[testdata.columns[chosen_features]])
        print("ACCURACY ON TRAIN SET :",accuracy_score(trainlabel,y_pred_ensemble_train))
        print("ACCURACY ON TEST SET",accuracy_score(testlabel,y_pred_ensemble))
        print(classification_report(testlabel,y_pred_ensemble))
        F1_scores_.append(f1_score(testlabel,y_pred_ensemble, average='macro'))
        print("CONFUSION_MATRIX")
        Cross_validate_GridSearch().printConfusionMatrix(testlabel,y_pred_ensemble,"Using Voting")
        Cross_validate_GridSearch().printROCplot(testlabel,y_pred_prob_ensemble,"Using Voting")
        print("MCC",matthews_corrcoef(testlabel, y_pred_ensemble))
        p2=Pipeline([("HandleMissingValues",HandleMissingValues()),("Normalize",Normalize()),("VarianceFilter",VarianceFilter()),("CorrelationFilter",Correlation_Filter()),('feature_sele',FeatureSelector()),("ENSEMBLE",ensemble)])   
        print("RESULT OF 5 fold Cross Validation")
        scoring = {'acc': 'accuracy','prec_macro': 'precision_macro','rec_micro': 'recall_macro'}
        scores = cross_validate(p2,trainset,trainlabel,scoring=scoring,cv=5, return_train_score=True)
        print(scores)
        print("Using LogisticRegression and GaussianNB without Smote")
        model1=LogisticRegression(random_state=50, solver='liblinear',penalty='l2',max_iter=100)
        model2=GaussianNB()
        Saving_Model.append(model1)
        Saving_Model.append(model2)
        print("Boruta Used For Feature Selection")
        F1_scores_LRGNB=Cross_validate_GridSearch().GNBLR(trainset,trainlabel,testset,testlabel)
        F1_scores_.append(F1_scores_LRGNB[0])
        F1_scores_.append(F1_scores_LRGNB[1])
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(Models_used,F1_scores_)
        figcount+=1
        Figureset.append(fig)
        plt.show()
        IndexOfBestModel=np.argmax(F1_scores_)
        Plotfigures().PlotAllfigures(Figureset,"Evaluate Different Models")
        return chosen_features,Saving_Model[IndexOfBestModel]

        

    def EvaluateDifferentModelswithPCA(self,trainset,trainlabel,testset,testlabel):
        global figcount
        global Figureset
        svm_gridsearch=self.SVM_GridSearch()
        rf_gridsearch=self.RandomForest_GridSearch()
        mlp_gridsearch=self.MLP_classifier_Gridsearch()
        grid_search_models=[svm_gridsearch,rf_gridsearch,mlp_gridsearch]
        best_models=[]
        Saving_Model=[]
        F1_scores_=[]
        Models_used=["SVM","RF","MLP","VC","LR","GNB"]
        k=-1
        for grid_search_model in grid_search_models:
            k=k+1
            pipeline  = Pipeline([("HandleMissingValues",HandleMissingValues()),("Normalize",Normalize()),("VarianceFilter",VarianceFilter()),("CorrelationFilter",Correlation_Filter()),('feature_sele',FeatureSelector()),("PCA",pca()),('clf_cv',grid_search_model)])
            pipeline.fit(trainset,trainlabel)
            chosen_features=pipeline.named_steps['feature_sele'].get_feature_names()
            chosen_model=pipeline.named_steps['clf_cv'].best_estimator_
            p2=Pipeline([("HandleMissingValues",HandleMissingValues()),("Normalize",Normalize()),("VarianceFilter",VarianceFilter()),("CorrelationFilter",Correlation_Filter()),('feature_sele',FeatureSelector()),("PCA",pca()),("model",pipeline.named_steps['clf_cv'].best_estimator_)])
            best_models.append(("model"+str(k),chosen_model))
            Saving_Model.append(pipeline.named_steps['clf_cv'].best_estimator_) # Appending best estimator
            y_pred_train=pipeline.predict(trainset)
            print("ACCURACY ON TRAIN SET :",accuracy_score(trainlabel,y_pred_train))
            y_pred=pipeline.predict(testset)
            y_prob=pipeline.predict_proba(testset)
            print("The important Features are :",chosen_features)
            print("Classification Report for TEST SET Using ",chosen_model)
            F1_scores_.append(f1_score(testlabel,y_pred, average='macro'))
            print(classification_report(testlabel,y_pred))
            print("ACCURACY ON TEST SET :",accuracy_score(testlabel,y_pred))
            print("CONFUSION_MATRIX")
            Cross_validate_GridSearch().printConfusionMatrix(testlabel,y_pred,Models_used[k])
            Cross_validate_GridSearch().printROCplot(testlabel,y_prob,Models_used[k])
            print("MCC",matthews_corrcoef(testlabel, y_pred))
            print("__________________________________________________________________________________________")
            print("RESULT OF 5 fold Cross Validation")
            scoring = {'acc': 'accuracy','prec_macro': 'precision_macro','rec_micro': 'recall_macro'}
            scores = cross_validate(p2,trainset,trainlabel,scoring=scoring,cv=5, return_train_score=True)
            print(scores)
        print("USING A VOTING ENSEMBLE WITH THE ABOVE BEST CLASSIFIERS")
        ensemble = VotingClassifier(best_models,n_jobs=-1,voting='soft')
        traindata,testdata=Preprocess_Data().normalize(trainset,testset)
        Saving_Model.append(ensemble) # Appending Voting model
        ensemble.fit(traindata[traindata.columns[chosen_features]],trainlabel)
        y_pred_ensemble=ensemble.predict(testdata[testdata.columns[chosen_features]])
        y_pred_ensemble_train=ensemble.predict(traindata[traindata.columns[chosen_features]])
        y_pred_prob_ensemble=ensemble.predict_proba(testdata[testdata.columns[chosen_features]])
        print("ACCURACY ON TRAIN SET :",accuracy_score(trainlabel,y_pred_ensemble_train))
        print("ACCURACY ON TEST SET",accuracy_score(testlabel,y_pred_ensemble))
        print(classification_report(testlabel,y_pred_ensemble))
        print("CONFUSION_MATRIX")
        Cross_validate_GridSearch().printConfusionMatrix(testlabel,y_pred_ensemble,"Using Voting with PCA")
        Cross_validate_GridSearch().printROCplot(testlabel,y_pred_prob_ensemble,"Using Voting with PCA")
        print("MCC",matthews_corrcoef(testlabel, y_pred_ensemble))
        F1_scores_.append(f1_score(testlabel,y_pred_ensemble, average='macro'))
        print("RESULT OF 5 fold Cross Validation")
        p2=Pipeline([("HandleMissingValues",HandleMissingValues()),("Normalize",Normalize()),("VarianceFilter",VarianceFilter()),("CorrelationFilter",Correlation_Filter()),('feature_sele',FeatureSelector()),("PCA",pca()),("ENSEMBLE",ensemble)])
        scoring = {'acc': 'accuracy','prec_macro': 'precision_macro','rec_micro': 'recall_macro'}
        scores = cross_validate(p2,trainset,trainlabel,scoring=scoring,cv=5, return_train_score=True)
        print(scores)
        print("Using LogisticRegression and GaussianNB without Smote")
        print("Boruta Used For Feature Selection")
        model1=LogisticRegression(random_state=50, solver='liblinear',penalty='l2',max_iter=100)
        model2=GaussianNB()
        Saving_Model.append(model1)
        Saving_Model.append(model2)
        F1_score_GNBLR=Cross_validate_GridSearch().GNBLR(trainset,trainlabel,testset,testlabel)
        F1_scores_.append(F1_scores_LRGNB[0])
        F1_scores_.append(F1_scores_LRGNB[1])
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(Models_used,F1_scores_)
        figcount+=1
        Figureset.append(fig)
        plt.show()
        Figureset.append(fig)
        figcount+=1
        IndexOfBestModel=np.argmax(F1_scores_)
        Plotfigures().PlotAllfigures(Figureset,"Evaluate Different Models with PCA")
        return chosen_features,Saving_Model[IndexOfBestModel]
        
   

   
class Novel_Predictions:

    def Mergealldata(self,data,data_labels,Novel_data):
        Novel_data_=Novel_data[data.columns]
        X_train,X_test=Preprocess_Data().Process_data(self,data,Novel_data)
        return X_train,X_test

    def Predict_Novelligands(self,model,X_train,X_test,y_train,selected_features):
        model.fit(X_train[selected_features],y_train)
        pred_labels=model.predict(X_test[selected_features])
        prob=model.predict_proba(X_test[selected_features])
        return pred_labels,prob

    def Predict_Novelligands_withSmote(self,model,X_train,X_test,y_train,selected_features,ORname):
        oversampler= sv.MSMOTE(proportion=0.80,random_state=50)
        X_samp, y_samp= oversampler.sample(X_train.values,y_train.values)
        X_samp_df=pd.DataFrame(X_samp)
        y_samp_df=pd.DataFrame(y_samp)
        X_samp_df.columns=X_test.columns
        model.fit(X_samp_df[selected_features],y_samp)
        pred_labels=model.predict(X_test[selected_features])
        try:
            prob=model.predict_proba(X_test[selected_features])
        except:
            print("This model does not support probability scores")
            prob=[]
        Novel_ligands_predictions=pd.DataFrame(list(zip(pred_labels,prob,X_test["Ligand"])),columns =['Predicted Label','Predicted probability','Ligand'])
        Novel_ligands_predictions.to_csv("Predictions_Unknown"+ORname)
        print("Saved file........."+"Predictions_Unknown"+ORname)
        return pred_labels,prob

    
class Plot_data:
 
    def TSNE_plot(self,data,data_labels):
        tsne = TSNE(n_components=2, random_state=50)
        transformed_data = tsne.fit_transform(data)
        k = np.array(transformed_data)
        Group=["Class 0","Class 1"]
        plt.scatter(k[:, 0],k[:, 1], c=data_labels)
        plt.legend(loc="lower right")
        plt.show()

class HandleDataImbalance:
    def Smote(self,traindata,trainlabel,prop):
        oversampler= sv.MSMOTE(proportion=prop,random_state=50)
        X_samp, y_samp= oversampler.sample(traindata.values,trainlabel.values)     
        Plot_data().TSNE_plot(X_samp, y_samp)
        X_samp= pd.DataFrame(X_samp)
        y_samp=pd.DataFrame(y_samp)
        X_samp.columns =list(traindata.columns.values)
        return X_samp,y_samp

    def EvaluatePerformanceSmote(self,trainset,trainlabel,testset,testlabel):
        global figcount
        global Figureset
        svm_gridsearch=Cross_validate_GridSearch().SVM_GridSearch()
        rf_gridsearch=Cross_validate_GridSearch().RandomForest_GridSearch()
        mlp_gridsearch=Cross_validate_GridSearch().MLP_classifier_Gridsearch()
        grid_search_models=[svm_gridsearch,rf_gridsearch,mlp_gridsearch]
        best_models=[]
        F1_scores_=[]
        Saving_Model=[]
        Models_used=["SVM","RF","MLP","VC","LR","GNB"]
        k=-1
        for grid_search_model in grid_search_models:
            k=k+1
            pipeline  = sample_pipeline([("HandleMissingValues",HandleMissingValues()),("sampling",SMOTE(random_state=50)),("Normalize",Normalize()),("VarianceFilter",VarianceFilter()),("CorrelationFilter",Correlation_Filter()),('feature_sele',FeatureSelector()),("PCA",pca()),('clf_cv',grid_search_model)])
            pipeline.fit(trainset,trainlabel)
            chosen_features=pipeline.named_steps['feature_sele'].get_feature_names()
            chosen_model=pipeline.named_steps['clf_cv'].best_estimator_
            p2= sample_pipeline([("HandleMissingValues",HandleMissingValues()),("sampling",SMOTE(random_state=50)),("Normalize",Normalize()),("VarianceFilter",VarianceFilter()),("CorrelationFilter",Correlation_Filter()),('feature_sele',FeatureSelector()),("PCA",pca()),("model",pipeline.named_steps['clf_cv'].best_estimator_)])
            best_models.append(("model"+str(k),chosen_model))
            Saving_Model.append(pipeline.named_steps['clf_cv'].best_estimator_) # Appending best estimator
            y_pred_train=pipeline.predict(trainset)
            print("ACCURACY ON TRAIN SET :",accuracy_score(trainlabel,y_pred_train))
            y_pred=pipeline.predict(testset)
            y_prob=pipeline.predict_proba(testset)
            print("The important Features are :",chosen_features)
            print("Classification Report for TEST SET Using ",chosen_model)
            print(classification_report(testlabel,y_pred))
            F1_scores_.append(f1_score(testlabel,y_pred, average='macro'))
            print("ACCURACY ON TEST SET :",accuracy_score(testlabel,y_pred))
            print("CONFUSION_MATRIX")
            Cross_validate_GridSearch().printConfusionMatrix(testlabel,y_pred,Models_used[k])
            Cross_validate_GridSearch().printROCplot(testlabel,y_prob,Models_used[k])
            print("MCC",matthews_corrcoef(testlabel, y_pred))
            print("__________________________________________________________________________________________")
            print("RESULT OF 5 fold Cross Validation")
            scoring = {'acc': 'accuracy','prec_macro': 'precision_macro','rec_micro': 'recall_macro'}
            scores = cross_validate(p2,trainset,trainlabel,scoring=scoring,cv=5, return_train_score=True)
            print(scores)
        print("USING A VOTING ENSEMBLE WITH THE ABOVE BEST CLASSIFIERS")
        ensemble = VotingClassifier(best_models,n_jobs=-1,voting='soft')
        sm = SMOTE(random_state=50)
        X_res, y_res = sm.fit_resample(trainset, trainlabel)
        X_res=pd.DataFrame(X_res)
        X_res.columns =list(trainset.columns.values)
        traindata,testdata=Preprocess_Data().normalize(X_res,testset)
        Saving_Model.append(ensemble) # Appending Voting model
        ensemble.fit(traindata[traindata.columns[chosen_features]],y_res)
        y_pred_ensemble=ensemble.predict(testdata[testdata.columns[chosen_features]])
        y_pred_ensemble_train=ensemble.predict(traindata[traindata.columns[chosen_features]])
        y_pred_prob_ensemble=ensemble.predict_proba(testdata[testdata.columns[chosen_features]])
        print("ACCURACY ON TRAIN SET :",accuracy_score(y_res,y_pred_ensemble_train))
        print("ACCURACY ON TEST SET",accuracy_score(testlabel,y_pred_ensemble))
        print(classification_report(testlabel,y_pred_ensemble))
        F1_scores_.append(f1_score(testlabel,y_pred_ensemble, average='macro'))
        print("CONFUSION_MATRIX")
        Cross_validate_GridSearch().printConfusionMatrix(testlabel,y_pred_ensemble,"Using VotingClassifier")
        Cross_validate_GridSearch().printROCplot(testlabel,y_pred_prob_ensemble,"Using VotingClassifier")
        print("MCC",matthews_corrcoef(testlabel, y_pred_ensemble))
        print("RESULT OF 5 fold Cross Validation")
        p2=sample_pipeline([("HandleMissingValues",HandleMissingValues()),("sampling",SMOTE(random_state=50)),("Normalize",Normalize()),("VarianceFilter",VarianceFilter()),("CorrelationFilter",Correlation_Filter()),('feature_sele',FeatureSelector()),("PCA",pca()),("ENSEMBLE",ensemble)])
        scoring = {'acc': 'accuracy','prec_macro': 'precision_macro','rec_micro': 'recall_macro'}
        scores = cross_validate(p2,trainset,trainlabel,scoring=scoring,cv=5, return_train_score=True)
        print(scores)
        print("Using LogisticRegression and GaussianNB with Smote")
        print("Boruta Used For Feature Selection")
        model1=LogisticRegression(random_state=50, solver='liblinear',penalty='l2',max_iter=100)
        model2=GaussianNB()
        Saving_Model.append(model1)
        Saving_Model.append(model2)
        Plotfigures().PlotAllfigures(Figureset,"Evaluate Different Models with Smote")
        F1_score_GNBLR=Cross_validate_GridSearch().GNBLR_smote(trainset,trainlabel,testset,testlabel)
        F1_scores_.append(F1_scores_GNBLR[0])
        F1_scores_.append(F1_scores_GNBLR[1])
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(Models_used,F1_scores_)
        Figureset.append(fig)
        figcount+=1
	plt.show()
	IndexOfBestModel=np.argmax(F1_scores_)
        Plotfigures().PlotAllfigures(Figureset,"Evaluate Different Models with Smote")
        plt.show()
        return chosen_features,Saving_Model[IndexOfBestModel]


class Plotfigures:
    def PlotAllfigures(self,figlist,Name):
        global ORName
        pdf = matplotlib.backends.backend_pdf.PdfPages(Name+ORName+".pdf")
        for i in range(len(figlist)):
            pdf.savefig(figlist[i])
        pdf.close()

def TakeInput(filepath,hmdb_filepath,OR_name):
    positive_Cancer=extractPositiveOnes(filepath)
    data_hmdb=pd.read_csv(hmdb_filepath,encoding="ISO-8859-1")
    positive_Cancer=pd.read_csv(filepath,encoding="ISO-8859-1")
    hmdb_names=data_hmdb['NAME']
    hmdb_SMILES=data_hmdb['SMILES']
    positive_Cancer_SMILES=positive_Cancer['Smiles']
    positive_Cancer_Names=positive_Cancer["Ligand"]
    hmdb_data=pd.concat([hmdb_SMILES,hmdb_names],axis=1)
    dataframe=pd.concat([positive_Cancer_SMILES,positive_Cancer_Names],axis=1)
    Cancer_clean_data=dataframe.drop_duplicates()
    Cancer_clean_data=Cancer_clean_data.reset_index(drop=True)
    df1= pd.DataFrame({"Cancer_Molecule":[],"Cancer_SMILES":[],"HMDB_Molecule":[],"HMDB_SMILES":[],"TANIMOTO_Similarity_Value":[]}) 
    hmdb_data=hmdb_data.reset_index(drop=True)
    k=0
    for i in range(len(Cancer_clean_data)):
        # df1=df1.iloc[0:0]
        # df1= pd.DataFrame({"Cancer_clean_data_Molecule":[],"Cancer_clean_data_SMILES":[],"HMDB_Molecule":[],"HMDB_SMILES":[],"TANIMOTO_Similarity_Value":[]}) 
        y=Chem.MolFromSmiles(Cancer_clean_data['Smiles'][i])
        fps1=FingerprintMols.FingerprintMol(y)
        for j in range(len(hmdb_data)):
            try:
                x=Chem.MolFromSmiles(hmdb_data['SMILES'][j])
                fps2=FingerprintMols.FingerprintMol(x)
                sim_val=DataStructs.FingerprintSimilarity(fps1,fps2)
                if sim_val>=0.85:# threshold for similarity value
                    df1.loc[k]=[Cancer_clean_data['Ligand'][i],Cancer_clean_data['Smiles'][i],hmdb_data['NAME'][j],hmdb_data['SMILES'][j],sim_val]
                    k=k+1
            except:
                print("WARNING")
        print("Comparison Done for Ligand :"+str(i))        
    df1.to_csv("Final_test_set_"+OR_name+".csv") 
    Ligand=df1["Cancer_clean_data_Molecule"]
    Smiles=df1["Cancer_clean_data_SMILES"]
    Activation_Status=[]
    Shortlisted_Metabolites=pd.DataFrame(list(zip(Smiles,Ligand,Activation_Status)),columns =['Smiles','Ligand','Activation Status'])
    Shortlisted_Metabolites=Shortlisted_Metabolites.drop_duplicates(subset='Ligand',keep='first')
    Shortlisted_Metabolites.to_csv("Shortlisted_Metabolites"+OR_Name+".csv")
    print("Shortlisted_Metabolites"+OR_Name+".csv"+" has been saved")
    print("Congrats! Final_test_set_"+OR_Name+".csv has been successfully saved!")

def extractPositiveOnes(FullFile):
    fullfile=pd.read_csv(FullFile)
    Smiles=[]
    Ligand=[]
    for i in range(len(fullfile)):
        if fullfile['Activation Status'][i]==1:
            Smiles.append(fullfile['Smiles'][i])
            Ligand.append(fullfile['Ligand'][i])
    positive_data = pd.DataFrame(list(zip(Smiles,Ligand)),columns =['Smiles','Ligand'])
    return positive_data

