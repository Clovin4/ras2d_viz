import numpy as np
import xarray as xr
# from sklearn.metrics import r2_score,mean_absolute_error
g = 'geometry'

def NSE(y_true,y_sim,dim=None):
    dimm = {'dim':dim} if dim else {}
    return 1-( ((y_true-y_sim)**2).sum(**dimm) / ((y_true-np.mean(y_true))**2).sum(**dimm) )

def MAE(y_true,y_sim,dim=None):
    dimm = {'dim':dim} if dim else {}
    return np.abs(y_true-y_sim).mean(**dimm)

# def R2(y_true,y_sim,dim=None):
#     #This link has the R2 formula identical to NSE? 
#     # https://en.wikipedia.org/wiki/Coefficient_of_determination
#     # Moriaseli et al 2007 gives no formula
#     dimm = {'dim':dim} if dim else {}

def RMSE(y_true,y_sim,dim=None,normed=False):
    ''' ## Root Mean Squared Error\n
    \mathrm{RMSD} = \sqrt{\frac{\sum_{i=1}^{N}\left(x_{i}-\hat{x}_{i}\right)^{2}}{N}}\n\n
    - y_true	=	actual observations time series
    - y_sim	=	estimated time series\n
    - N	=	number of non-missing data points
    y_true,y_sim: xr da's or np arrays of true/observed values vs simulated/predicted\n
    normed: whether to normalize to percentage of y_true:\n
    RMSE*( N / y_sim.sum() ) \n\n
    dim='t', named xr da dim to aggregate along
    '''
    dimm = {'dim':dim} if dim else {}
    rms = np.sqrt(np.square( y_sim - y_true ).mean(**dimm))
    if normed:
        # rms = rms*y_true.count(**dimm)/y_true.sum(**dimm)
        rms = rms/np.abs(y_true.mean(**dimm))
        
    return rms

def PBIAS(y_true,y_sim,dim=None):
    ''' ## Percent Bias\n
    \mathrm{PBIAS} = 100\left[\frac{\sum_{i=1}^{N}\left(x_{i}-\hat{x}_{i}\right)}{\sum_{i=1}^{N}x_{i}}\right]\n\n
    - y_true	=	actual observations time series
    - y_sim	=	estimated time series\n
    - N	=	number of non-missing data points
    y_true,y_sim: xr da's or np arrays of true/observed values vs simulated/predicted\n
    dim='t', named xr da dim to aggregate along
    '''
    dimm = {'dim':dim} if dim else {}
    return 100*( (y_true-y_sim).sum(**dimm) / y_true.sum(**dimm) )

def R2(y_true,y_sim,dim=None):
    '''
    The R2 metric calculates the coefficient of determination for the given input arrays "y_true" and "y_sim". The function returns the R2 value for the two input arrays.
        - y_true: actual observations time series
        - y_sim: estimated time series
        - dim: named xr da dim to aggregate along. 
               It is "None" by default. 
    '''
    dimm = {'dim':dim} if dim else {}
    return 1 - ( ((y_true-y_sim)**2).sum(**dimm) / ((y_true-np.mean(y_true))**2).sum(**dimm) )


def PCC(y_true,y_sim,dim=None):
    ''' ## Pearson Correlation Coefficient\n
    \mathrm{PCC} = \frac{\sum_{i=1}^{N}\left(x_{i}-\bar{x}\right)\left(\hat{x}_{i}-\bar{\hat{x}}\right)}{\sqrt{\sum_{i=1}^{N}\left(x_{i}-\bar{x}\right)^2}\sqrt{\sum_{i=1}^{N}\left(\hat{x}_{i}-\bar{\hat{x}}\right)^2}}\n\n
    - y_true	=	actual observations time series
    - y_sim	=	estimated time series\n
    - N	=	number of non-missing data points
    y_true,y_sim: xr da's or np arrays of true/observed values vs simulated/predicted\n
    dim='t', named xr da dim to aggregate along
    '''
    dimm = {'dim':dim} if dim else {}
    return ( ((y_true-np.mean(y_true))*(y_sim-np.mean(y_sim))).sum(**dimm) / np.sqrt( ((y_true-np.mean(y_true))**2).sum(**dimm) * ((y_sim-np.mean(y_sim))**2).sum(**dimm) ) )
