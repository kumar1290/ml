import pandas as pd
import numpy as np


def isConsistent(drow, g, cols):
    k = 0
    for c in range(len(cols)-1):
        if g[c] == "?":
            k += 1
            continue
        elif g[c] != drow[k]:
            return False
        k += 1

    return True

# Generating Version Space


def getVersionSpace(s, g):
    versionSpace = []
    for i in range(len(g)):
        x, y = s.copy(), g[i].copy()
        for j in range(len(s)):
            if x[j] == y[j]:
                continue
            elif y[j] == "?" and s[j] != "?":
                x[j] = "?"
                y[j] = s[j]
                versionSpace.append(x)
                versionSpace.append(y)
                break
    # versionSpace= list(set(versionSpace))
    return versionSpace


def CandidateElimination(df):
    noOfCols = len(df.columns)-1
    s = ["null" for i in range(1, noOfCols+1)]
    g = [["?" for i in range(1, noOfCols+1)]]
    print("Initial most specific hypothesis  : ", s)
    for i in range(len(df)):
        # print("r: ",df.loc[i,:])
        r = df.loc[i, :]
        k = 0
        if r['EnjoySport'] == "Yes":
            for j in df.columns:
                for x in g:
                    if isConsistent(r, x, df.columns) == False:
                        g.remove(x)
                if j == "EnjoySport":
                    continue
                # print("K : ",r[j], hypothesis[k], r[j]==hypothesis[k] )
                if r[j] == s[k]:
                    k += 1
                    continue
                elif s[k] == "null":
                    s[k] = r[j]
                elif r[j] != s[k]:
                    s[k] = "?"
                k += 1
        else:
            newG = []
            for x in g:
                k = 0
                # t= x.copy()
                for j in df.columns:
                    if j == "EnjoySport":
                        continue
                    if r[j] != s[k] and s[k] != "?":
                        tem = x.copy()
                        tem[k] = s[k]
                        newG.append(tem)
                    k += 1
            g.clear()
            g = newG
        # print("Specific : ",s)
        # print("Final hypothesis : ",g)
        # print(hypothesis)
    vSpace = getVersionSpace(s[:], g[:])
    # versionSpace= [*set(vSpace)]
    versionSpace = []
    for x in vSpace:
        if x not in versionSpace:
            versionSpace.append(x)
    return s, g, versionSpace


def getOutput(versionSpace, y):
    res = True
    # y= y.re
    for h in versionSpace:
        for i in range(len(h)):
            if h[i] == '?' or h[i] == y[i]:
                continue
            else:
                res = False
                break
        if res:
            return "Yes"
    
    return "No"

def performaceAnalysis(versionSpace,data):
    tp,tn,fp,fn=0,0,0,0
    for i in range(len(data)):
        tem= np.array(df.loc[[i],:])
        last = len(tem[0])
        predOutput = getOutput(versionSpace,tem[0][0:last-1])
        actOutput = data.iloc[i,len(data.columns)-1]
        print(predOutput,"  ",actOutput)
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

if __name__ == "__main__":
    df = pd.read_csv("ML\datasets\EnjoySport.csv")
    print(df)
    s, g, versionSpace = CandidateElimination(df)
    print("Final most specific hypothesis : ", s)
    print("Final most general hypothesis : ", g)
    print("Version Space: ", versionSpace)
    performaceAnalysis(versionSpace,df)
