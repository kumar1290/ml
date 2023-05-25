import pandas as pd
import numpy as np
import math

class FindS :

    def __init__(self , X , labels,finalClass) :
        self.X= X
        self.finalClass= finalClass
        self.labels = labels
        self.hypothesis =[ "null" for i in range(len(X.columns))]
        self.labelCategories = list(set(labels))
        self.labelCategoriesCount = len(set(labels)) 

    def trainModel(self):
        noOfCols= len(self.X.columns)
        # hypothesis =[ "null" for i in range(1,noOfCols+1)]
        # print("Initial most specific hypothesis  : ", hypothesis)
        for i in range(len(self.X)) :
            # print("r: ",df.loc[i,:])
            r= self.X.loc[i,:]
            k=0
            if self.labels[i]=="Yes" :
                for j in self.X.columns:
                    # if j=="EnjoySport" :
                    #     continue
                    # print("K : ",r[j], hypothesis[k], r[j]==hypothesis[k] )
                    if r[j] == self.hypothesis[k] :
                        k+=1
                        continue
                    elif self.hypothesis[k]=="null" :
                        self.hypothesis[k]= r[j]
                    elif r[j]!=self.hypothesis[k]:
                        self.hypothesis[k]= "?"
                    k+=1
            # print("Hypothesis : ", self.hypothesis)

    def displayFinalHypothesis(self) :
        print("Final Hypothesis : ",self.hypothesis) 


    def getOutput(self,y):
        res= True
        # y= y.re
        for i in range(len(self.hypothesis)):
            if self.hypothesis[i]=='?' or self.hypothesis[i]==y[i]:
                continue
            else :
                res=False
                break

        if res:
            return "Yes"
        else :
            return "No"


    def performaceAnalysis(self,data):
        tp,tn,fp,fn=0,0,0,0
        for i in range(len(data)):
            tem= np.array(df.loc[[i],:])
            last = len(tem[0])
            predOutput = self.getOutput(tem[0][0:last-1])
            actOutput = data.iloc[i,len(data.columns)-1]
            if actOutput == "Yes":
                if predOutput=="Yes":
                    tp= tp+1
                else :
                    fn= fn+1
            else :
                if predOutput=="No":
                    tn= tn+1
                else :
                    fp= fp+1
        print("Performance Analysis : ")
        accuracy = (tp+tn)/(tp+fp+tn+fn)*100
        print("Accuracy : ",accuracy)
        precision = (tp)/(tp+fp)*100
        print("Precision : ",precision)
        recall =(tp)/(tp+fn)*100
        print("Recall : ", recall)
        f1Score = (2*precision*recall)/(precision+recall)

        print("F1 Score : ", f1Score)
        

if __name__=="__main__":
    df= pd.read_csv("ML\datasets\heart.csv")
    print(df.loc[0:10,:])
    noOfRows= len(df)
    # print(noOfRows)
    cols= df.columns
    fclass=cols[len(cols)-1]
    x= df.drop([fclass], axis=1).loc[0:math.floor(0.5*noOfRows),:]
    findSModel= FindS(X=x,labels=df[fclass].copy(),finalClass=fclass)
    findSModel.trainModel()
    findSModel.displayFinalHypothesis()
    findSModel.performaceAnalysis(df.loc[math.floor(0.5*noOfRows):, :])
    