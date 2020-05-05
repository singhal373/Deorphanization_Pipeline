import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
from rdkit import rdBase
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Fingerprints import FingerprintMols
import pandas as pd


def TakeInput(filepath,hmdb_filepath,OR_name):
    positive_Cancer=extractPositiveOnes(filepath)
    data_hmdb=pd.read_csv(hmdb_filepath,encoding="ISO-8859-1")
    hmdb_names=data_hmdb['NAME']
    hmdb_SMILES=data_hmdb['SMILES']
    positive_Cancer_SMILES=positive_Cancer['Smiles']
    positive_Cancer_Names=positive_Cancer["Ligand"]
    hmdb_data=pd.concat([hmdb_SMILES,hmdb_names],axis=1)
    dataframe=pd.concat([positive_Cancer_SMILES,positive_Cancer_Names],axis=1)
    Cancer_clean_data=dataframe.drop_duplicates()
    Cancer_clean_data=Cancer_clean_data.reset_index(drop=True)
    df1= pd.DataFrame({"Test_Molecule":[],"Test_SMILES":[],"HMDB_Molecule":[],"HMDB_SMILES":[],"TANIMOTO_Similarity_Value":[]}) 
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
                if sim_val>=0.80:# threshold for similarity value
                    df1.loc[k]=[Cancer_clean_data['Ligand'][i],Cancer_clean_data['Smiles'][i],hmdb_data['NAME'][j],hmdb_data['SMILES'][j],sim_val]
                    k=k+1
            except:
                print("WARNING")
        print("Comparison Done for Ligand :"+str(i))        
    df1.to_csv("Final_test_set_"+OR_name+".csv") 
    Ligand=df1["HMDB_Molecule"]
    Smiles=df1["HMDB_SMILES"]
    Activation_Status=['?'] * len(Ligand)
    Shortlisted_Metabolites=pd.DataFrame(list(zip(Smiles,Ligand,Activation_Status)),columns =['Smiles','Ligand','Activation Status'])
    Shortlisted_Metabolites=Shortlisted_Metabolites.drop_duplicates(subset='Ligand',keep='first')
    Shortlisted_Metabolites.to_csv("Shortlisted_Metabolites"+OR_name+".csv")
    print("Shortlisted_Metabolites"+OR_name+".csv"+" has been saved")
    print("Congrats! Final_test_set_"+OR_name+".csv has been successfully saved!")

import pandas
def extractPositiveOnes(FullFile):
    fullfile=pandas.read_csv(FullFile,engine='python')
    Smiles=[]
    Ligand=[]
    for i in range(len(fullfile)):
        if fullfile['Activation Status'][i]==1:
            Smiles.append(fullfile['Smiles'][i])
            Ligand.append(fullfile['Ligand'][i])
    positive_data = pandas.DataFrame(list(zip(Smiles,Ligand)),columns =['Smiles','Ligand'])
    return positive_data




    		



