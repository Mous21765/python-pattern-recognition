import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class housing_price:
    def __init__(self,dataset='housing.csv'):
        self.Dataset=pd.read_csv(dataset)
        #REMOVING NaN VALUES FROM THE DATASET
        self.Dataset=self.Dataset.dropna()
        self.Dataset=self.Dataset.reset_index()
        scaler=MinMaxScaler()
        scaled=pd.DataFrame()
        self.Column_Names=self.Dataset.columns
        #SCALING NUMERICAL COLUMNS
        for i in range(1,len(self.Column_Names)-1):
            n=self.Column_Names[i]
            scaled[n]=pd.DataFrame(scaler.fit_transform(self.Dataset[n].to_numpy().reshape(-1,1)))
        #ONE HOT ENCODE CATEGORICAL COLUMN
        scaled[self.Column_Names[-1]]=self.Dataset[self.Column_Names[-1]]
        scaled=scaled.sort_values(by=[self.Column_Names[-1]])
        scaled=scaled.reset_index()
        new_columns=scaled[self.Column_Names[-1]].unique().tolist()
        ohe=pd.DataFrame(OneHotEncoder().fit_transform(scaled[[self.Column_Names[-1]]]).toarray())
        ohe.columns=new_columns
        for i in new_columns:
            scaled[i]=ohe[i]
        scaled=scaled.drop('ocean_proximity',axis=1)
        self.Scaled_Dataset=scaled


    def visualise_data(self):
        #HISTOGRAM CREATION
        plt.figure(figsize=(20,10))
        for i in range(1,len(self.Column_Names)-1):
            n=self.Column_Names[i]
            plt.subplot(2,5,i)
            plt.hist(self.Dataset[n])
            plt.title(n)
        plt.savefig('histograms')
        plt.show()
        #VISUALASIATION GRAPHS
        plt.figure(figsize=(10,10))
        plt.title('Households according to Population')
        plt.scatter(self.Dataset['households'],self.Dataset['population'],s=1)
        plt.xlabel('Households')
        plt.ylabel('Population')
        plt.savefig('housepop')
        plt.show()
        plt.figure(figsize=(10,10))
        plt.title('House Value according to Median Income')
        plt.scatter(self.Dataset['median_house_value'],self.Dataset['median_income'],s=0.8)
        plt.xlabel('Median House Value')
        plt.ylabel('Median Income')
        plt.savefig('incomeprice')
        plt.show()
        plt.figure(figsize=(10,10))
        plt.title('Total Rooms per Population')
        plt.xlabel('Total Rooms')
        plt.ylabel('Population')
        plt.scatter(self.Dataset['total_rooms'],self.Dataset['population'],s=0.8)
        plt.savefig('roomspop')
        plt.show()
       
    def create_dataset(self):
        scores=self.Scaled_Dataset['median_house_value'].to_numpy()
        columns=self.Scaled_Dataset.columns.tolist()
        columns.remove('median_house_value')
        dataset_values=pd.DataFrame()
        for i in range(1,len(columns)):
            dataset_values[columns[i]]=self.Scaled_Dataset[columns[i]]
        dataset_values=dataset_values.to_numpy()
        self.dataset_values=dataset_values
        mean=np.mean(scores)
        flags=[]
        for i in scores:
            if(i>mean):
                flags.append(1)
            else:
                flags.append(-1)
        self.scores=scores
        self.flags=flags
        
    def perceptron_weight_change(self,wt,X,y,rt):
        #ALGORITHM BASED ON PAGE 102 ON PATTERN RECOGNITION BOOK
        Y=[]
        dx=[]
        X=np.array(X)
        y=np.array(y)
        for i in range(0,len(X)):
            if((y[i]*np.dot(wt,X[i]))<0):
                Y.append(X[i])
                dx.append(y[i]*-1)
        J=dx[0]*Y[0]
        for i in range(1,len(Y)):
            J=J+dx[i]*Y[i]
        wt=np.array(wt)
        wt1=wt-rt*J
        return wt1

    def perceptron(self):
        X=self.dataset_values
        y=self.flags
        w=np.random.rand(X.shape[1])
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.1)
        #10 TRAINING EPOCHS
        for i in range(0,10):
            w=self.perceptron_weight_change(w,xtrain,ytrain,0.05)
        y_model=[]
        for i in range(0,len(xtest)):
            y_model.append(np.sign(np.dot(w,xtest[i])))
        mse=mean_squared_error(ytest,y_model)
        mae=mean_absolute_error(ytest,y_model)
        print('Mean Absolute Error:',mae)
        print('Mean Squared Error:',mse)
        print('Weight Vector:',w)

    def least_squares(self):
        #BASED ON ALGORITHM OF BOOK PATTERN RECOGNITION 118
        X=self.dataset_values
        y=self.flags
        xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.1)
        xtx=np.dot(np.transpose(xtrain),xtrain)
        xty=np.dot(np.transpose(xtrain),ytrain)
        w=np.dot(np.linalg.inv(xtx),xty)
        y_model=[]
        for i in range(0,len(xtest)):
            y_model.append(np.sign(np.dot(w,xtest[i])))
        mae=mean_absolute_error(ytest,y_model)
        mse=mean_squared_error(ytest,y_model)
        print('Mean Absolute Error:',mae)
        print('Mean Squared Error:',mse)
        print('Weight Vector:',w)


    def regression(self):
        xtrain,xtest,ytrain,ytest=train_test_split(self.dataset_values,self.scores)
        reg=LinearRegression().fit(xtrain,ytrain)
        print('Regression Score:',reg.score(xtrain,ytrain))
        y_model=[]
        for i in xtest:
            y_model.append(reg.predict(i.reshape(1,-1)))
        mae=mean_absolute_error(ytest,y_model)
        mse=mean_squared_error(ytest,y_model)
        print('Mean Absolute Error:',mae)
        print('Mean Squared Error:',mse)

