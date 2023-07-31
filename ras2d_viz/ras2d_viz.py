import os, h5py
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
if __package__ is None or __package__ == '':
    # uses current directory visibility
    import metrics
    # from helper import *
else:
    # uses current package visibility
    from . import metrics
    # from .helper import *

import SHPshifter as shp
g = 'geometry'
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# import pydsstools.heclib as hec

# import holoviews as hv
# hv.extension('plotly')

# model class defines the HEC RAS model itself and will be used to retreive information from the model
class Model():
    def __init__(self, mod_loc) -> None:
        self.mod_loc = mod_loc

        # self.hdfFile.close()
    def getSimulations(self, gageVec, gageDataCSV):
        hdf_dict = self.getSimLocations()
        sims = {sim: Simulation(hdf_dict[sim], gageVec, gageDataCSV) for sim in hdf_dict}
        return sims

    def getSimLocations(self):
        hdf_files = list(Path(self.mod_loc).rglob('*.hdf'))
        hdf_dict = {self.getSimInfo(hdf)['Plan ShortID']: hdf for hdf in hdf_files}
        return hdf_dict
    
    
    def getSimInfo(self, hdfPath):
        # TODO: get the plan and runtime info from the model
        self.hdfFile = h5py.File(hdfPath, 'r')
        planGen = self.hdfFile['Plan Data']['Plan Information'].attrs
        planGen = pd.Series(planGen).map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        # close hdf file
        self.hdfFile.close()
        return planGen
    
    def getSimParams(self, hdfPath):
        self.hdfFile = h5py.File(hdfPath, 'r')
        planParams = self.hdfFile['Plan Data']['Plan Parameters'].attrs
        planParams = pd.Series(planParams).map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        # close hdf file
        self.hdfFile.close()
        return planParams
    
    def getSimVolAccounting(self, hdfPath):
        self.hdfFile = h5py.File(hdfPath, 'r')
        DDQAs = pd.DataFrame(np.array(self.hdfFile['Geometry/2D Flow Areas/Attributes']))['Name'].str.decode('utf-8').to_list()
        if len(DDQAs)>1:
            raise NotImplementedError(f'{len(DDQAs)}: {DDQAs}\n2D flow areas found in {self.pHDFPath}. Only 1 2DQA is supported')
        flowArea=DDQAs[0]
        volAcct = self.hdfFile['Results/Unsteady/Summary/Volume Accounting/Volume Accounting 2D'][flowArea].attrs
        volAcct = pd.Series(volAcct).map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        # close hdf file
        self.hdfFile.close()
        return volAcct

# simulation class defines the simulation and will be used to retreive information from the simulation, this class inherits from the model class and will use the model class to retreive information from the model
class Simulation():
    def __init__(self, hdfPath, gageVec, gageDataCSV, crs = 'EPSG:6479') -> None:
        self.crs = crs
        self.hdfPath = hdfPath
        self.gageDataCSV = gageDataCSV
        self.gageVec = gageVec

        self.getCellPts()
        self.getStage()

        self.hdfFile.close()
        
    def getCellPts(self):
        self.hdfFile = h5py.File(self.hdfPath, 'r')
        gages = gpd.read_file(self.gageVec)
        if gages.crs != self.crs:
            print('Gage vector is not in the correct crs, converting to crs: {}'.format(self.crs))
            gages = gages.to_crs(self.crs)

        assert 'SITE_ID' in gages.columns, \
            f'ERROR:\nGauge needs to be a column in {self.gageVec}. Found {gages.columns}'
        
        self.gage_names = dict(zip(gages.SITE_ID, gages.SITENAME))
        
        gages = gages[['SITE_ID', 'geometry']]
        gages = gages.set_index('SITE_ID')

        obsData = self.gageDataCSV
        cols = pd.read_csv(obsData).set_index('datetime').columns
        gages = gages[gages.index.isin(cols)]

        self.hdfFile = h5py.File(self.hdfPath, 'r')
        # get the area name
        DQAlvl = self.hdfFile['Geometry']['2D Flow Areas']
        DDQAs = pd.DataFrame(np.array(self.hdfFile['Geometry/2D Flow Areas/Attributes']))['Name'].str.decode('utf-8').to_list()
        if len(DDQAs)>1:
            raise NotImplementedError(f'{len(DDQAs)}: {DDQAs}\n2D flow areas found in {self.pHDFPath}. Only 1 2DQA is supported')
        self.flowArea=DDQAs[0]
        hdfCells=DQAlvl[self.flowArea]['Cells Center Coordinate']

        # creates dataframe from cell locations based on hdf file
        cells=pd.DataFrame(np.array(hdfCells))

        # renames columns to be easily identified
        cells=cells.rename(columns={0:'X',1:'Y'})

        # creates column with appropriate cell numbers
        cells['Cell']=range(0,len(cells))

        # creates geodataframe from dataframe using coordinates
        cells=gpd.GeoDataFrame(cells,geometry=gpd.points_from_xy(cells.X,cells.Y,crs=self.crs))

        self.cells = cells
        self.gages = gages

    def getStage(self):
        
        # get the area name
        planInfo = self.hdfFile['Plan Data']['Plan Information'].attrs

        # convert the plan information to a pandas series
        self.planInfo= pd.Series(planInfo).map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        self.planID = self.planInfo['Plan ShortID']

        self.gageDataObs = pd.read_csv(self.gageDataCSV)
        self.gageDataObs.set_index('datetime', inplace=True)
        self.gageDataObs.index = pd.to_datetime(self.gageDataObs.index)

        simStage = self.hdfFile['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas'][self.flowArea]['Water Surface']
        self.simStage = pd.DataFrame(np.array(simStage))

        ts = pd.Series(np.array(self.hdfFile['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['Time Date Stamp']))
        ts = ts.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        tsteps = pd.to_datetime(ts, format='%d%b%Y %H:%M:%S')
        tsteps.name = 'time'
        # add to the dataframe
        self.simStage = self.simStage.set_index(tsteps)
        self.simStage.index.name = 'time'
        #give the column row a name
        self.simStage.columns.name = 'Cell'

        self.simStage = self.simStage

         # find simulated stage at gauge points by inverse distance weighted interpolation

        # While WSEL data in RAS is stored at cell centroids, when plotting a timeseries in
        # RASmapper or viewing an inundation layer, RAS uses a similar multi point interpolation
        # in realtime. Good interpolation parameters with a close approximation to RAS are 4 
        # nearest neighbors (ideal for square cells) whose distance is weighted by the power
        #  of -2 (k=4,p=2). 

        wsel = xr.DataArray(self.simStage).to_dataset(name='WSEL')

        celldf = self.cells.set_index('Cell').drop(g, axis=1)

        wsel = wsel.assign_coords(x=celldf.X, y=celldf.Y)['WSEL']

        self.gageDataSim = shp.nd.IDW(self.gages, wsel, k=4, p=2)
        self.gageDataSim = self.gageDataSim.to_pandas()

        # ensure that the time steps are the same for the simulated and observed data
        dtObs = self.gageDataObs.index[1]-self.gageDataObs.index[0]

        # resample the simulated data to match the observed data
        self.gageDataSim = self.gageDataSim.resample(dtObs).mean()

        dtSim = self.gageDataSim.index[1]-self.gageDataSim.index[0]

        if dtSim>dtObs:
            print(f'WARNING: Simulation time step is larger than observed time step.\nSim: {dtSim}\nObs: {dtObs}')
            self.gageDataObs = self.gageDataObs.resample(dtSim).mean()
        elif dtSim<dtObs:
            print(f'WARNING: Simulation time step is smaller than observed time step.\nSim: {dtSim}\nObs: {dtObs}')
            self.gageDataSim = self.gageDataSim.resample(dtObs).mean()
            
        common = set(self.gageDataSim.index).intersection(set(self.gageDataObs.index))

        self.gageDataSim = self.gageDataSim.loc[common].sort_index()
        self.gageDataObs = self.gageDataObs.loc[common].sort_index()
        
        assert self.gageDataSim.index.equals(self.gageDataObs.index), 'The simulated and observed time series do not have the same time steps'

    def getMetrics(self):
        # return a dataframe of the metrics for each gage
        self.metrics = pd.DataFrame(index=self.gageDataSim.columns, columns=['MAE', 'RMSE', 'PCC'])
        for gage in self.gageDataSim.columns:
            # self.metrics.loc[gage, 'NSE'] = metrics.NSE(self.gageDataObs[gage], self.gageDataSim[gage])
            self.metrics.loc[gage, 'MAE'] = metrics.MAE(self.gageDataObs[gage], self.gageDataSim[gage])
            self.metrics.loc[gage, 'RMSE'] = metrics.RMSE(self.gageDataObs[gage], self.gageDataSim[gage], normed=True)
            # self.metrics.loc[gage, 'PBIAS'] = metrics.PBIAS(self.gageDataObs[gage], self.gageDataSim[gage])
            # self.metrics.loc[gage, 'R2'] = metrics.R2(self.gageDataObs[gage], self.gageDataSim[gage])
            self.metrics.loc[gage, 'PCC'] = metrics.PCC(self.gageDataObs[gage], self.gageDataSim[gage])
        return self.metrics
    
    def getMetrics_Comp(self, comp : 'Simulation'):
        # return a dataframe of the metrics for each gage
        # given another simulation object, return the metrics for each gage between the two simulations
        self.metrics = pd.DataFrame(index=self.gageDataSim.columns, columns=['MAE', 'RMSE', 'PCC'])
        for gage in self.gageDataSim.columns:
            # self.metrics.loc[gage, 'NSE'] = metrics.NSE(comp.gageDataSim[gage], self.gageDataSim[gage])
            self.metrics.loc[gage, 'MAE'] = metrics.MAE(comp.gageDataSim[gage], self.gageDataSim[gage])
            self.metrics.loc[gage, 'RMSE'] = metrics.RMSE(comp.gageDataSim[gage], self.gageDataSim[gage], normed=True)
            self.metrics.loc[gage, 'PBIAS'] = metrics.PBIAS(comp.gageDataSim[gage], self.gageDataSim[gage])
            # self.metrics.loc[gage, 'R2'] = metrics.R2(comp.gageDataSim[gage], self.gageDataSim[gage])
            self.metrics.loc[gage, 'PCC'] = metrics.PCC(comp.gageDataSim[gage], self.gageDataSim[gage])
        return self.metrics
    
    # def plotGages_hv_Comp(self, comp: 'Simulation', gages=None, save=False, saveLoc=None):
    #     if gages is None:
    #         gages = self.gageDataSim.columns
    #     elif isinstance(gages, str):
    #         gages = [gages]

    #     h_map = hv.HoloMap({gage: hv.Curve(self.gageDataSim[gage], label='Simulated') * hv.Curve(self.gageDataObs[gage], label='Observed') * hv.Curve(comp.gageDataSim[gage], label='Simulated Comp') for gage in gages}, kdims='Gage')
    #     h_map.opts(title='Gage Simulated vs Observed', xlabel='Time', ylabel='Stage (ft)', width=800, height=400)
    #     # # add the metrics to the plot
    #     # for gage in gages:
    #     #     h_map[gage].opts(title=f'Gage {gage} Simulated vs Observed\nNSE: {self.metrics.loc[gage, "NSE"]:.3f}\nMAE: {self.metrics.loc[gage, "MAE"]:.3f}\nRMSE: {self.metrics.loc[gage, "RMSE"]:.3f}')

    #     if save:
    #         hv.save(h_map, saveLoc)

    #     return h_map
    
    def plotGages_Comp(self, comp: 'Simulation', gages=None, save=False, saveLoc=None):
        # return variable names of self and comp
        s = self.planID
        c = comp.planID
        # plot the simulated data for each gage for the current simulation and the comparison simulation
        # return a plotly figure of each gage simulated vs observed along with the metrics
        # each gage is to have a seperate subplot
        if gages is None:
            gages = self.gageDataSim.columns
        elif isinstance(gages, str):
            gages = [gages]
        elif isinstance(gages, list):
            pass
        else:
            raise TypeError('gages must be a string or list of strings')
        
        # check that the comp is a valid comparison
        assert isinstance(comp, Simulation), 'comp must be a Simulation object'

        gs = list(self.gageDataObs.columns)
        mets = pd.DataFrame(index=gs, columns=['Gage', 'MAE', 'RMSE', 'PBIAS', 'PCC'])
        mets.Gage = gs
        # mets.NSE = metrics.NSE(self.gageDataObs, self.gageDataSim)
        mets.MAE = metrics.MAE(self.gageDataObs, self.gageDataSim)
        mets.RMSE = metrics.RMSE(self.gageDataObs, self.gageDataSim)
        # mets.PBIAS = metrics.PBIAS(self.gageDataObs, self.gageDataSim)
        # mets.R2 = metrics.R2(self.gageDataObs, self.gageDataSim)
        mets.PCC = metrics.PCC(self.gageDataObs, self.gageDataSim)

        metsComp = pd.DataFrame(index=gs, columns=['Gage', 'MAE', 'RMSE', 'PBIAS', 'PCC'])
        metsComp.Gage = gs
        # metsComp.NSE = metrics.NSE(self.gageDataObs, comp.gageDataSim)
        metsComp.MAE = metrics.MAE(self.gageDataObs, comp.gageDataSim)
        metsComp.RMSE = metrics.RMSE(self.gageDataObs, comp.gageDataSim)
        # metsComp.PBIAS = metrics.PBIAS(self.gageDataObs, comp.gageDataSim)
        # metsComp.R2 = metrics.R2(self.gageDataObs, comp.gageDataSim)
        metsComp.PCC = metrics.PCC(self.gageDataObs, comp.gageDataSim)

        fig = make_subplots(rows=len(self.gageDataObs.columns), cols=2,
                            column_widths=[0.7, 0.3],
                            vertical_spacing=len(self.gageDataObs.columns)/2500,
                            horizontal_spacing=0.02,
                            subplot_titles=['Simulated vs Observed Metrics']*len(self.gageDataObs.columns)*2,
                            specs=[[{"type": "scatter"}, {"type": "table"}]]*len(self.gageDataObs.columns)
                        )
        
        for i, gage in enumerate(self.gageDataObs.columns):
            fig.layout.annotations[i]['font'] = dict(size=10)
            fig.add_trace(go.Scatter(x=self.gageDataObs.index, y=self.gageDataObs[gage], name='Observed', mode='lines', line=dict(color='blue'), showlegend=False), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=self.gageDataSim.index, y=self.gageDataSim[gage], name=f'{s}', mode='lines', line=dict(color='red'), showlegend=False), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=self.gageDataSim.index, y=comp.gageDataSim[gage], name=f'{c}', mode='lines', line=dict(color='green'), showlegend=False), row=i+1, col=1)
            fig.update_xaxes(
                linecolor='navy',
                title_text="Time (UTC)",
                row=i+1, col=1
            )
            fig.update_yaxes(
                linecolor='navy',
                title_text="Stage (ft)",
                row=i+1, col=1
            )
            fig.add_trace(go.Table(
                header=dict(
                    values=["Gage", 'MAE', 'RMSE', 'PBIAS', 'PCC'],
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[[f'{mets.Gage[i]}:{s}', f'{metsComp.Gage[i]}:{c}'], [round(mets.MAE[i],2), round(metsComp.MAE[i],2)], [round(mets.RMSE[i],2), round(metsComp.RMSE[i],2)], [round(mets.PBIAS[i],2), round(metsComp.PBIAS[i],2)], [round(mets.PCC[i],2), round(metsComp.PCC[i],2)]],
                    )
                ), row=i+1, col=2)
            # title each subplot with the gage name and number and add a horizontal line to separate each subplot

            fig.layout.annotations[i*2]['text'] = f'{gage}: {self.gage_names[gage]}'

            fig.update_layout(title='Simulated vs Observed', width=1400, height=400*len(self.gageDataObs.columns))

        if save:
            fig.write_html(saveLoc)

        return fig

    
    # def plotGages_hv(self, gages=None, save=False, saveLoc=None):
    #     if gages is None:
    #         gages = self.gageDataSim.columns
    #     elif isinstance(gages, str):
    #         gages = [gages]
    #     elif isinstance(gages, list):
    #         pass

    #     h_map = hv.HoloMap({gage: hv.Curve(self.gageDataSim[gage], label='Simulated') * hv.Curve(self.gageDataObs[gage], label='Observed') for gage in gages}, kdims='Gage')
    #     h_map.opts(title='Gage Simulated vs Observed', xlabel='Time', ylabel='Stage (ft)', width=800, height=400)
    #     # # add the metrics to the plot
    #     # for gage in gages:
    #     #     h_map[gage].opts(title=f'Gage {gage} Simulated vs Observed\nNSE: {self.metrics.loc[gage, "NSE"]:.3f}\nMAE: {self.metrics.loc[gage, "MAE"]:.3f}\nRMSE: {self.metrics.loc[gage, "RMSE"]:.3f}')

    #     if save:
    #         hv.save(h_map, saveLoc)

    #     return h_map

            
    def plotGages(self, gages=None, save=False, saveLoc=None):
        # return a plotly figure of each gage simulated vs observed along with the metrics
        # each gage is to have a seperate subplot
        if gages is None:
            gages = self.gageDataSim.columns
        elif isinstance(gages, str):
            gages = [gages]
        elif isinstance(gages, list):
            pass
        else:
            raise TypeError('gages must be a string or list of strings')
        
        gs = list(self.gageDataObs.columns)
        mets = pd.DataFrame(index=gages, columns=['Gage', 'MAE', 'RMSE', 'PBIAS', 'PCC'])
        mets.Gage = gs
        # mets.NSE = metrics.NSE(self.gageDataObs, self.gageDataSim)
        mets.MAE = metrics.MAE(self.gageDataObs, self.gageDataSim)
        mets.RMSE = metrics.RMSE(self.gageDataObs, self.gageDataSim)
        # mets.PBIAS = metrics.PBIAS(self.gageDataObs, self.gageDataSim)
        # mets.R2 = metrics.R2(self.gageDataObs, self.gageDataSim)
        mets.PCC = metrics.PCC(self.gageDataObs, self.gageDataSim)
        
        fig = make_subplots(rows=len(self.gageDataObs.columns), cols=2,
                            column_widths=[0.7, 0.3],
                            vertical_spacing=len(self.gageDataObs.columns)/2500,
                            horizontal_spacing=0.02,
                            subplot_titles=['Simulated vs Observed Metrics']*len(self.gageDataObs.columns)*2,
                            specs=[[{"type": "scatter"}, {"type": "table"}]]*len(self.gageDataObs.columns)
                        )
        
        for i, gage in enumerate(self.gageDataObs.columns):
            fig.layout.annotations[i]['font'] = dict(size=10)
            fig.add_trace(go.Scatter(x=self.gageDataObs.index, y=self.gageDataObs[gage], name='Observed', mode='lines', line=dict(color='blue'), showlegend=False), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=self.gageDataSim.index, y=self.gageDataSim[gage], name='Simulated', mode='lines', line=dict(color='red'), showlegend=False), row=i+1, col=1)
            fig.update_xaxes(
                linecolor='navy',
                title_text="Time (UTC)",
                row=i+1, col=1
            )
            fig.update_yaxes(
                linecolor='navy',
                title_text="Stage (ft)",
                row=i+1, col=1
            )
            fig.add_trace(go.Table(
                header=dict(
                    values=["Gage", 'MAE', 'RMSE', 'PBIAS', 'PCC'],
                    font=dict(size=10),
                    align="left"
                ),
                cells=dict(
                    values=[mets.Gage[i], round(mets.MAE[i],2), round(mets.RMSE[i],2), round(mets.PCC[i],2)],
                    align = "left",
                    # color NSE, MAE, and RMSE cells based on the value
                    font_color = [
                        ['black'],
                        # ['green' if mets.NSE[i]>0 else 'red'],
                        ['green' if mets.MAE[i]<1 else 'red'],
                        ['green' if mets.RMSE[i]<15 else 'red'],
                        ['green' if mets.PBIAS[i]>10 else 'red'],
                        # ['green' if abs(mets.R2[i])<0.9 else 'red'],
                        ['green' if mets.PCC[i]>0.9 else 'red']
                    ]
                    )
                ), row=i+1, col=2)
                # title each subplot with the gage name and number and add a horizontal line to separate each subplot
            fig.layout.annotations[i*2]['text'] = f'{gage}: {self.gage_names[gage]}'
            fig.update_layout(title='Simulated vs Observed', width=1400, height=400*len(self.gageDataObs.columns))

        if save:
            fig.write_html(saveLoc)

        return fig
    
    def plotGages_plt(self, comp, save=False, saveLoc=None):
        # return a matplotlib figure of each gage simulated vs observed along with the metrics
        # each gage is an individual plot that is returned or saved to a file
        # return variable names of self and comp
        s = self.planID
        c = comp.planID

        # check that the comp is a valid comparison
        assert isinstance(comp, Simulation), 'comp must be a Simulation object'

        for gage in self.gageDataObs.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.gageDataObs[gage], label='Observed')
            ax.plot(self.gageDataSim[gage], label=f'{s}')
            ax.plot(comp.gageDataSim[gage], label=f'{c}')
            ax.set_title(f'{gage}: {self.gage_names[gage]}')
            ax.set_xlabel('Time (UTC)')
            ax.set_ylabel('Stage (ft)')
            ax.legend()
            if save:
                fig.savefig(f'{saveLoc}/{gage}.png', dpi=300)
            else:
                return fig

        return fig

# class HMS_Model():
#     def __init__(self, hms_dir) -> None:
#         self.hms_dir = hms_dir

#     def getSimulations(self, keyword=None):
#         if keyword is None:
#             dss_files = list(Path(self.hms_dir).glob('*.dss'))
#         else:
#             dss_files = list(Path(self.hms_dir).glob(f'*{keyword}*.dss'))

#         dss_dict = {dss.stem.split("_")[0]: dss for dss in dss_files}

#         return dss_dict
        