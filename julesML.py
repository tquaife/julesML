from datetime import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.neural_network import MLPRegressor as nnr
from sklearn import svm

from julesDataHandler import *


if __name__=="__main__":
 
    #load data
    #target=julesData("data/gh_point_smc_avail_top.txt")    
    #target=julesData("data/gh_point_smcl_lvl1.txt")
    target=julesData("data/gh_point_smcl_lvl2.txt")
    feature1=julesData("data/gh_point_t1p5m_gb.txt")
    feature1.transform_data_to_day_of_year()
    feature2=julesData("data/gh_point_t1p5m_gb.txt")
    feature2.transform_data_rc_running_mean()
    feature3=julesData("data/gh_point_precip.txt")
    feature3.transform_data_rc_running_mean()
    feature4=julesData("data/gh_point_precip.txt")
    feature4.lags=[0,1,2,3,4,5]
    
    feat_list=[feature1,feature2,feature3,feature4]
    
    #generate training and validation data
    n=len(target.data[:])
    sample=np.random.randint(365,n-365,10000)
    s_train=sampleBuilder(sample,target,feat_list)
    s_train.scale()
    #validation:
    sample=np.random.randint(365,n-365,10000)
    s_evalt=sampleBuilder(sample,target,feat_list)
    s_evalt.scale()
    #final year hindcast:
    sample=np.arange(n-365,n)
    s_hcast=sampleBuilder(sample,target,feat_list)
    s_hcast.scale()


    #SVM
    #n.b. data needs scaling.
    m=svm.NuSVR(kernel="rbf", C=1.0)
    
    #Use the SVM to generate a first order estimate of SM
    m.fit(s_train.X,s_train.Y)
    pred_evalt=m.predict(s_evalt.X)
    pred_train=m.predict(s_train.X)
    pred_hcast=m.predict(s_hcast.X)
    
    
    do_scatter_plot=False
    #do_scatter_plot=True    
    if do_scatter_plot:
        plt.plot(s_evalt.Y,pred_evalt,".")
        plt.plot(s_train.Y,pred_train,".")
    else:
        #rescale the data
        a=s_hcast.Y_mean
        b=s_hcast.Y_std
        plt.plot(s_hcast.Y*b+a,"-",label="JULES")
        plt.plot(pred_hcast*b+a,"-",label="ML")
        plt.xlabel('Day of year')
        plt.ylabel(s_hcast.target.var_name)
        plt.legend()
    plt.show()
    #plt.savefig("julesML_soil_moisture_example.png")
    
    
