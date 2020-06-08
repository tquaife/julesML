from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import ExtraTreesRegressor as etr
from sklearn.neural_network import MLPRegressor as nnr
from sklearn import svm
from datetime import datetime
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
data['smcl_lvl2']=None

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

def mk_training_sample(X,Y,nsamps=5000,lags=None, leave_at_end=365):      
    """Extract a random training set and include lagged variables.
    The leave_at_end parameter prevents the end of the time series
    being sampled so it can be used for validation (e.g. hindcasts)
    """
    (n,m)=np.shape(X)
    if lags==None:
        lags=numpy.zeros(n)
    nlags=sum(lags)+n
    train_exp=np.zeros([nlags,nsamps])
    train_obs=np.zeros(nsamps)
    for i in range(nsamps):
        #random int excluding the last year
        #and leaving enough room for all lags
        j=np.random.randint(max(lags),m-leave_at_end)
        train_obs[i]=Y[j]
        p=0
        for nn in range(n):
            for (k,lag) in enumerate(np.arange(lags[nn]+1)):
                train_exp[p,i]=X[nn,j-lag]
                p+=1         
    return train_obs, np.transpose(train_exp)    


def final_year(X,Y,ndays=365,lags=None):      
    """Extract the final year of the data.
    Used for hindcasts.
    """
    (n,m)=np.shape(X)
    if lags==None:
        lags=numpy.zeros(n)
    nlags=sum(lags)+n
    train_exp=np.zeros([nlags,ndays])
    train_obs=np.zeros(ndays)
    for i in range(ndays):
        j=len(Y)-ndays+i
        train_obs[i]=Y[j]
        p=0
        for nn in range(n):
            for (k,lag) in enumerate(np.arange(lags[nn]+1)):
                train_exp[p,i]=X[nn,j-lag]
                p+=1
    return train_obs, np.transpose(train_exp)    

def smooth_obs(obs,wsize=30):
    """Basic low pass filter
    """
    obs_smooth=np.zeros(np.shape(obs))
    for i in range(len(obs)):
        n=0
        for j in range(min(i,wsize)+1):
            n+=1
            obs_smooth[i]+=obs[i-j]
        obs_smooth[i]/=float(n)
    #This line centres the window:    
    #obs_smooth=np.roll(obs_smooth,int(-wsize/2.))    
    return obs_smooth

def scale_all(obs,expl):
    
    obs_mean=np.mean(obs)
    obs_std=np.std(obs)
    obs=(obs-obs_mean)/obs_std
    
    (n,m)=np.shape(expl)
    
    for i in range(m):
        mean=np.mean(expl[:,i])
        std=np.std(expl[:,i])
        expl[:,i]=(expl[:,i]-mean)/std
                   
    return obs, expl, obs_mean, obs_std


def include_day_of_year():
    """Generates a new entry in the data
    dictionary to hold the day of year.
    Assumes 'precip' has been loaded
    """
    data['day_of_year']={}
    data['day_of_year']['value']=[]
    data['day_of_year']['date']=[]
    for date_str in data['precip']['date'][:]:        
        d=datetime.strptime(date_str,"%Y-%m-%d")
        data['day_of_year']['date'].append(date_str)        
        data['day_of_year']['value'].append(d.timetuple().tm_yday)


def add_var_to_data(name,newvar):
    """
    """
    data[name]={}
    data[name]['value']=[]
    for datum in newvar:
        data[name]['value'].append(datum)


if __name__=="__main__":
 
    import matplotlib.pyplot as plt
    import sys

    #load data and make day of year a variable 
    load_data()
    include_day_of_year()

    #make a big matrix of all the forcing variables
    obs_var_name='smcl_lvl1'
    X=mk_forcing_matrix(data,['day_of_year'])
    Y=np.array(data[obs_var_name]['value'])
    
    lags=[0]
    (obs_train,explntry_train)=mk_training_sample(X,Y,nsamps=10000,lags=lags)
    (obs_hcast,explntry_hcast)=final_year(X,Y,ndays=len(Y),lags=lags)
    #print(explntry_train[:5,:])
    #print(obs_train[:5])
    #sys.exit()

    #SVM
    #n.b. data needs scaling.
    m=svm.NuSVR(kernel="rbf", C=1.0)
    
    #scale everything
    if True:
        obs_train,explntry_train,obs_train_mean,obs_train_std=scale_all(obs_train,explntry_train)
        #obs_test,explntry_test,obs_test_mean,obs_test_std=scale_all(obs_test,explntry_test)
        obs_hcast,explntry_hcast,obs_hcast_mean,obs_hcast_std=scale_all(obs_hcast,explntry_hcast)
    
    #Use the SVM to generate a first order estimate of SM
    m.fit(explntry_train,obs_train)
    pred=m.predict(explntry_hcast)
    add_var_to_data('sm_lvl1_doy',pred*obs_train_std+obs_train_mean)
    

    #plt.plot(pred[0:365],"-")
    #plt.show()
    
    
    #-----------------------------------------------------
    #Now do ML regression using the first pass sm estimate
    #-----------------------------------------------------
    
    obs_var_name='smcl_lvl1'
    X=mk_forcing_matrix(data,['sm_lvl1_doy','precip','t1p5m_gb'])
    Y=np.array(data[obs_var_name]['value'])
    
    lags=[0,60,45]
    (obs_train,explntry_train)=mk_training_sample(X,Y,nsamps=10000,lags=lags)
    (obs_test,explntry_test)=mk_training_sample(X,Y,lags=lags)
    (obs_hcast,explntry_hcast)=final_year(X,Y,lags=lags)


    #random forest
    m=rfr(n_estimators=200,bootstrap=True,random_state=0,min_samples_leaf=1,criterion="mse")

    #Fitting:
    m.fit(explntry_train,obs_train)
    pred_train=m.predict(explntry_train)
    pred_test=m.predict(explntry_test)
    pred_hcast=m.predict(explntry_hcast)
 
 
    
    do_scatter_plot=False
    #do_scatter_plot=True    
    if do_scatter_plot:
        plt.plot(obs_test,pred_test,".")
        plt.plot(obs_train,pred_train,".")
    else:
        plt.plot(obs_hcast,"-",label="JULES")
        plt.plot(explntry_hcast[:,0],"-",label="SV doy pred")
        plt.plot(pred_hcast,"-",label="ML")
        plt.xlabel('Day of year')
        plt.ylabel(obs_var_name)
        plt.legend()
    plt.show()
    #plt.savefig("soil_moisture_ml_example.png")
    
    
