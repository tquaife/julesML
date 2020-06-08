from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.neural_network import MLPRegressor as nnr
from sklearn import svm
import numpy as np
import pandas as pd

data={}
data['gpp_gb']=None
data['npp_gb']=None
data['fsmc_gb']=None
data['smc_avail_top']=None
data['precip']=None
data['t1p5m_gb']=None
data['sw_down']=None
data['lw_down']=None
data['q1p5m_gb']=None
data['smcl_lvl1']=None

def load_data():
    """Load the data files
    """
    for var_name in data:
        fn="gh_point_"+var_name+".txt"
        data[var_name]=pd.read_csv(fn, delim_whitespace=True)
        data[var_name]=column_cleaning(data[var_name])

def column_cleaning(frame):
    """Header line starts with a "#" on its own which
    causes a few problems. This cleans that up.
    """
    frame.columns = np.roll(frame.columns, len(frame.columns)-1)
    return frame.dropna(how='all', axis=1)

def mk_forcing_matrix(data,var_list):
    """Put a list of variables into a
    single matrix. These are the "forcing"
    or "explanatory" varilables
    """
    n=len(var_list)
    m=len(data[var_list[0]]['value'])
    forcing=np.zeros([n,m])
    for (i,v) in enumerate(var_list):
        forcing[i,:]=data[v]['value']
    return forcing

def mk_training_sample(X,Y,nsamps=5000,lags=[0],leave_at_end=365):      
    """Extract a random training set and include lagged variables.
    The leave_at_end parameter prevents the end of the time series
    being sampled so it can be used for validation (e.g. hindcasts)
    """
    (n,m)=np.shape(X)
    nlags=len(lags)
    train_exp=np.zeros([n*nlags,nsamps])
    train_obs=np.zeros(nsamps)
    for i in range(nsamps):
        #random int excluding the last year
        #and leaving enough room for all lags
        j=np.random.randint(max(lags),m-leave_at_end)
        train_obs[i]=Y[j]
        for (k,lag) in enumerate(lags):
            for nn in range(n):
                #print(lag,"::",k*n+nn,i,"::",nn,j+lag)
                train_exp[k*n+nn,i]=X[nn,j-lag]         
    return train_obs, np.transpose(train_exp)    


def final_year(X,Y,ndays=365,lags=[0]):      
    """Extract the final year of the data.
    Used for hindcasts.
    """
    (n,m)=np.shape(X)
    nlags=len(lags)
    train_exp=np.zeros([n*nlags,ndays])
    train_obs=np.zeros(ndays)
    for i in range(ndays):
        j=len(Y)-ndays+i
        train_obs[i]=Y[j]
        for (k,lag) in enumerate(lags):
            for nn in range(n):
                train_exp[k*n+nn,i]=X[nn,j-lag]
         
    return train_obs, np.transpose(train_exp)    



         
if __name__=="__main__":
 
    import matplotlib.pyplot as plt
 
    load_data()
    #print(data['t1p5m_gb'].columns)
    #print(data['t1p5m_gb']['value'][:5])

    #obs_var_name='smc_avail_top'
    obs_var_name='smcl_lvl1'
    #X=mk_forcing_matrix(data,['precip','t1p5m_gb','sw_down','q1p5m_gb','lw_down'])
    X=mk_forcing_matrix(data,['precip','t1p5m_gb','sw_down'])
    Y=np.array(data[obs_var_name]['value'])
    
    lags=np.arange(5)
    (obs_train,explntry_train)=mk_training_sample(X,Y,lags=lags)
    (obs_test,explntry_test)=mk_training_sample(X,Y,lags=lags)
    (obs_hcast,explntry_hcast)=final_year(X,Y,lags=lags)


    ##ML algorithms:    

    #random forest
    m=rfr(n_estimators=200,bootstrap=True,random_state=0,min_samples_leaf=1,criterion="mse")

    #extra trees
    #m=etr(n_estimators=300,)
    
    #MLP NN
    #m=nnr(hidden_layer_sizes=(200,10))
    
    #SVM
    #n.b. data needs scaling.
    #m=svm.NuSVR()
    
    #Fitting:
    m.fit(explntry_train,obs_train)
    pred_train=m.predict(explntry_train)
    pred_test=m.predict(explntry_test)
    pred_hcast=m.predict(explntry_hcast)
    
    
    #plt.plot(obs_test,pred_test,".")
    #plt.plot(obs_train,pred_train,".")
    plt.plot(obs_hcast,"-",label="JULES")
    plt.plot(pred_hcast,"-",label="ML")
    plt.xlabel('Day of year')
    plt.ylabel(obs_var_name)
    plt.legend()
    plt.show()
    #plt.savefig("soil_moisture_ml_example.png")
    
    
