from padelpy import from_smiles
import pandas as pd
# do a pip install padelpy before running this
class CalculateDescriptors:
  def getChemicalDescriptors(self,data_file,destpath,filename):# datafile(csv file) destpath(where to save) # filename (name of file to save)
    test1=pd.read_csv(data_file) 
    for i in range(len(test1)):
      try:
        temp=test1["Smiles"][i]
        descriptors = from_smiles(temp)
      except RuntimeError:
        temp=test1["Smiles"][i]
        descriptors = from_smiles(temp,timeout=30)
      if i==0:
        df = pd.DataFrame(descriptors, columns=descriptors.keys(),index=[0])
      else:
        df1 = pd.DataFrame(descriptors, columns=descriptors.keys(),index=[i])
      if i is 1:
        ff=pd.concat([df,df1], axis=0)
      if i>1:
        ff=pd.concat([ff,df1],axis=0)
    ff=pd.concat([test1['Activation Status'],ff,test1['Ligand']],axis=1)      
    ff.to_csv(destpath+filename+".csv",index=False)



    