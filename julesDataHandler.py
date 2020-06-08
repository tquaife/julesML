from datetime import datetime
from copy import copy

import numpy as np


class julesData(object):

    def __init__(self, file_name):
        """Class to hold jules data and
        provide samples for machine learning
        """
        self.file_name=file_name
        self.read_data()
        self.lags=[0,]
        
    def read_data(self):        
        """read in the data & dates from the file
        """
        self.read_header()
        indx=self.header.index("value")
        self.data=np.genfromtxt(self.file_name,usecols=(indx,),skip_header=1)
        indx=self.header.index("date")
        self.dates=np.genfromtxt(self.file_name,usecols=(indx,),skip_header=1,dtype='U')
        
    def read_header(self):
        """read the header line of the file
        and get the variable name from the first
        line of data
        """
        with open(self.file_name) as f: 
            head=f.readline()
            data=f.readline()        
        self.header=head.split()[1:]
        self.var_name=data.split()[0]
    
    def transform_data_to_day_of_year(self):
        """replace the data with the day of 
        year. Useful for input into ML.
        """   
        self.var_name="day_of_year"  
        for (i,date_str) in enumerate(self.dates):
            self.data[i]=datetime.strptime(date_str,"%Y-%m-%d").timetuple().tm_yday
   
    def transform_data_rc_running_mean(self,window_size=30):
        """replace the data with a *right* centered running mean
        """
        filt=np.ones(window_size)
        data_new=copy(self.data)
        for i in range(len(self.data)):
            data_new[i]=np.mean(self.data[np.max([0,i-window_size+1]):i+1])        
        self.data=data_new
        
        
class sampleBuilder(object):

    def __init__(self, samples, target, feature_list):
        """Class to generate and hold training/testing
        data sets for use with ML algorithms.
        
        samples :: integer list/array of sample positions
        target  :: an instance of the julesData class holding target data
        feature_list :: a list of julesData class instances holding feature data
        """        
        
        self.samples=np.array(samples)
        self.target=target
        self.feature_list=feature_list
        self.is_scaled=False
        self.build_data()
        
    def build_data(self):    
        """Build the data that will go into the ML algorithm
        where X are the features and Y is the target variable.
        """
        
        #target variable:
        self.Y=self.target.data[self.samples]
        
        #count features (because some have
        #multiple lags):
        nfeat=0           
        for feature in self.feature_list:
            nfeat+=len(feature.lags)
        self.X=np.zeros([len(self.Y),nfeat])
        
        #build X, the feature matrix:
        nfeat=0    
        for feature in self.feature_list:
            for lag in feature.lags:
                self.X[:,nfeat]=feature.data[self.samples-lag]
                nfeat+=1      
        
    def scale(self):
        """Set the target variable and each feature so
        that it has mean=0 and var=1. Needed for some 
        ML techniques.
        """        
        #don't scale multiple times
        if self.is_scaled:
            return

        #need to store statistics so
        #we can unscale in needed
        self.Y_mean=self.Y.mean()
        self.Y_std=self.Y.std()
        self.Y=(self.Y-self.Y_mean)/self.Y_std        
        
        self.X_mean=self.X.mean(axis=0)
        self.X_std=self.X.std(axis=0)
        for i in range(np.shape(self.X)[1]):
            self.X[:,i]=(self.X[:,i]-self.X_mean[i])/self.X_std[i]        
        
        self.is_scaled=True
        
        
    def unscale(self):
        """Revert scaled data back to original values
        """        
        #don't unscale multiple times
        if self.is_scaled==False:
            return

        self.Y=self.Y*self.Y_std+self.Y_mean       
        for i in range(np.shape(self.X)[1]):
            self.X[:,i]=self.X[:,i]*self.X_std[i]+self.X_mean[i]
        
        self.is_scaled=False



        
if __name__=="__main__":
    """A few tests of the above code.
    """    

    target=julesData("data/gh_point_smc_avail_top.txt")
    
    feature1=julesData("data/gh_point_t1p5m_gb.txt")
    feature1.transform_data_to_day_of_year()
    
    feature2=julesData("data/gh_point_t1p5m_gb.txt")
    feature2.lags=[0,1,2]
    
    s=sampleBuilder([3,4,5,6,7,8,15,401],target,[feature1,feature2])
    print(s.X)
    print(s.X.mean(axis=0))    
    print(s.X.std(axis=0))    

    print("-----------------------")
    s.scale()
    print(s.target.data[:10])
    print(s.Y)
    print(s.X)
    print(s.X.mean(axis=0))    
    print(s.X.std(axis=0))    

    print("-----------------------")
    s.unscale()
    print(s.target.data[:10])
    print(s.Y)
    print(s.X)
    print(s.X.mean(axis=0))    
    print(s.X.std(axis=0))    


