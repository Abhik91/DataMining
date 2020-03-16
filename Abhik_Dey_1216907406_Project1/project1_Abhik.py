#!/usr/bin/env python
# coding: utf-8

# In[263]:


######################################################################################################################
# Project1
# Subject - CSE572 Data Mining
# Author - Abhik Dey (1216907406)
# Goal is:
#    1) Predict the meal timing of the patients so as to adinister insuling
#    2) Extract 4 features
#    3) Apply PCA and extract top 5 components
######################################################################################################################
import numpy as np
import pandas as pd
from scipy import stats
import scipy.fftpack
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[264]:


def interpolate_data(cgm_series_row_data,cgm_dt_row_data):
    #print(cgm_series_row_data)
    cgm_series_data_transpose = []
    cgm_dt_data_transpose = []
    glu =[]
    datetime = []
    
    for element in cgm_series_row_data:
        cgm_series_data_transpose.append(element)
    
    for element in cgm_dt_row_data:
        cgm_dt_data_transpose.append(element)   

    data = {'dt': cgm_dt_data_transpose, 'glucose':cgm_series_data_transpose}  
    df = pd.DataFrame(data)
    df['dt'].interpolate(inplace=True)
    datetime  = df['dt'].values.tolist()
    df.set_index('dt', inplace=True)
    df['newZVal'] = df['glucose'].interpolate(method = 'polynomial', order = 2)
    glu = df['newZVal'].values.tolist()
    return glu, datetime


# In[265]:


def clean_data():
    max_limit = 0.7 # Allowing 30% NaN data to interpolate, delete the row if >= 70%
    drop_row = []
    
    # Interpolate NaN if row has less NaN records than max_limit for cgm_series
    for i in range(len(df_glucose_pat)):
        no_of_nan_glucose = df_glucose_pat.iloc[i].isnull().sum()
        if no_of_nan_glucose > 0:
            percent_of_data = (no_of_nan_glucose/len(df_glucose_pat.iloc[i]))
            if percent_of_data < max_limit:
                df_glucose_pat.loc[i],df_date_num_pat.loc[i] = interpolate_data(df_glucose_pat.iloc[i],df_date_num_pat.iloc[i])


# In[266]:


def calFFT():
    x = []        
    df_glucose_pat['FFTPeak1'] = -1
    df_glucose_pat['FFTPeak2'] = -1
    
    
    for i in range(0,len(df_glucose_pat_cp.iloc[0])):
        x.append(i*5)
        
    for elem in range(len(df_glucose_pat)):
        cgm_fft_values  = abs(scipy.fftpack.fft(df_glucose_pat_cp.iloc[elem].values))
        #print (len(cgm_fft_values))
        #print (len(x))
        #cgm_dt_values = df_date_num_lunch_pat1.iloc[elem].values
        val = set(cgm_fft_values)
        val = sorted(val, reverse = True)
        #print (cgm_fft_values)
        #print (val)
        firstHighPeak = list(val)[1]
        secondHighPeak = list(val)[2]        
        df_glucose_pat['FFTPeak1'].iloc[elem] = firstHighPeak
        df_glucose_pat['FFTPeak2'].iloc[elem] = secondHighPeak   
        
        #print (cgm_fft_values)
        #print (firstHighPeak,secondHighPeak)
        plt.plot(x,cgm_fft_values)#,use_line_collection = True
        plt.ylim(0,1000)
        plt.show()


# In[267]:


#take maximum amplitude
#df_glucose_pat
def calWelch():
    df_glucose_pat['maxAmplitude'] = -1
    df_glucose_pat['stdAmplitude'] = -1
    df_glucose_pat['meanAmplitude'] = -1
    for elem in range(len(df_glucose_pat_cp)):
        v, welch_values  = np.array((signal.welch(df_glucose_pat_cp.iloc[elem].values)))
#         print (welch_values)
#         print (v)
#         print (welch_values.tolist().index(max(welch_values)))
#         print (max(np.sqrt(cgm_fft_values)))
#         print (np.where(cgm_fft_values == max(cgm_fft_values)))
        df_glucose_pat['maxAmplitude'].iloc[elem] = np.sqrt(max(welch_values))
        df_glucose_pat['stdAmplitude'].iloc[elem] = np.std(np.sqrt(welch_values))
        df_glucose_pat['meanAmplitude'].iloc[elem] = np.mean(np.sqrt(welch_values))
        
    
        plt.plot(v,np.sqrt(welch_values))#,use_line_collection = True
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.show()


# In[268]:


def calCgmVelocity():
    df_glucose_pat['meanCgmVel'] = -1
    df_glucose_pat['stdCgmVel'] = -1
    df_glucose_pat['medianCgmVel'] = -1
    
    time_interval = 10 # taking time interval as 10 mins    
    for elem in range(len(df_glucose_pat_cp)):
        #print ('**********************************************')
        #print ('Counter - ',elem)
        window_size = 2
        velocity = []
        row_data = df_glucose_pat_cp.iloc[elem].values
        row_length = len(row_data)
        #print (row_data)
        counter = 0
        x_cgmvel = []
        for i in range((len(row_data) - window_size)):
            x_cgmvel.append(counter)
            counter += 5
            disp = (row_data[i] - row_data[i + window_size])
            vel = disp / time_interval
            velocity.append(vel)
        df_glucose_pat['meanCgmVel'].iloc[elem] = np.mean(velocity)
        df_glucose_pat['stdCgmVel'].iloc[elem] = np.std(velocity)
        df_glucose_pat['medianCgmVel'].iloc[elem] = np.median(velocity)
        #rint (velocity)
        #print (x_cgmvel)
        plt.plot(x_cgmvel,velocity)#,use_line_collection = True
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.show()


# In[269]:


def calPolyfit():
    
    df_glucose_pat['polyCoeff1'] = -1
    df_glucose_pat['polyCoeff2'] = -1
    df_glucose_pat['polyCoeff3'] = -1
    df_glucose_pat['polyCoeff4'] = -1
    df_glucose_pat['polyCoeff5'] = -1
    df_glucose_pat['polyCoeff6'] = -1
    time_interval = [j * 5 for j in range(0, len(df_glucose_pat_cp.iloc[0]))]
    for elem in range(len(df_glucose_pat_cp)):
        polyfit = list(np.polyfit(time_interval, df_glucose_pat_cp.iloc[elem], 5))
#         print ("###########################################################")
#         print (len(polyfit))
#         print (polyfit)
        plt.plot(polyfit)
        plt.xlabel('Polynomial Order')
        plt.ylabel('Coefficient')
        plt.show()
        for i in range(len(polyfit)):
            col = 'polyCoeff'+str(i+1)
            df_glucose_pat[col].iloc[elem] = polyfit[i]


# In[270]:


def performPCA():
    features = feature_matrix.columns
    fm = feature_matrix.loc[:, features].values
    # Normalize the feature values.
    fm = stats.zscore(fm)
    pca_cons = PCA(n_components = 5)
    principal_components = pca_cons.fit_transform(fm)
    final_component = pd.DataFrame(data = principal_components, 
                                   columns = ['component_1', 'component_2','component_3','component_4','component_5'])
    
    print (sum(pca_cons.explained_variance_ratio_))
    print (final_component)
    
    #pca = PCA().fit(feature_matrix)
    x_axis =  ['PCA1','PCA2','PCA3','PCA4','PCA5']
    y_axis = pca_cons.explained_variance_ratio_
    plt.plot(np.cumsum(y_axis))
    plt.bar(x_axis,y_axis)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.show()
    
    #Spyder Plot
    columns = ['component_1', 'component_2','component_3','component_4','component_5']
    for i in range(len(final_component)):
        value = list(final_component.iloc[i])    
        value += value[:1]
        angles = [n / float(len(columns)) * 2 * np.pi for n in range(len(columns))]
        angles += angles[:1]
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1],columns)
        ax.plot(angles,value)
        ax.fill(angles, value, 'teal', alpha=0.1)
        labl = 'Timeseries_' + str(i)
        ax.set_title(labl)
        plt.show()
    
    return final_component


# In[256]:


if __name__ == '__main__':
        
    # Creating master dataframes to hold all the features and datapoints of all 5 patients
    df_date_num_pat = pd.DataFrame(columns = ['cgmDatenum_ 1', 'cgmDatenum_ 2', 'cgmDatenum_ 3', 'cgmDatenum_ 4',
           'cgmDatenum_ 5', 'cgmDatenum_ 6', 'cgmDatenum_ 7', 'cgmDatenum_ 8',
           'cgmDatenum_ 9', 'cgmDatenum_10', 'cgmDatenum_11', 'cgmDatenum_12',
           'cgmDatenum_13', 'cgmDatenum_14', 'cgmDatenum_15', 'cgmDatenum_16',
           'cgmDatenum_17', 'cgmDatenum_18', 'cgmDatenum_19', 'cgmDatenum_20',
           'cgmDatenum_21', 'cgmDatenum_22', 'cgmDatenum_23', 'cgmDatenum_24',
           'cgmDatenum_25', 'cgmDatenum_26', 'cgmDatenum_27', 'cgmDatenum_28',
           'cgmDatenum_29', 'cgmDatenum_30', 'cgmDatenum_31'])

    df_glucose_pat = pd.DataFrame(columns = ['cgmSeries_ 1', 'cgmSeries_ 2', 'cgmSeries_ 3', 'cgmSeries_ 4',
           'cgmSeries_ 5', 'cgmSeries_ 6', 'cgmSeries_ 7', 'cgmSeries_ 8',
           'cgmSeries_ 9', 'cgmSeries_10', 'cgmSeries_11', 'cgmSeries_12',
           'cgmSeries_13', 'cgmSeries_14', 'cgmSeries_15', 'cgmSeries_16',
           'cgmSeries_17', 'cgmSeries_18', 'cgmSeries_19', 'cgmSeries_20',
           'cgmSeries_21', 'cgmSeries_22', 'cgmSeries_23', 'cgmSeries_24',
           'cgmSeries_25', 'cgmSeries_26', 'cgmSeries_27', 'cgmSeries_28',
           'cgmSeries_29', 'cgmSeries_30', 'cgmSeries_31'])

    for i in range(5):
        df_date_num_lunch_pat = pd.read_csv('DataFolder/CGMDatenumLunchPat'+str(i+1)+'.csv')
        df_series_lunch_pat = pd.read_csv('DataFolder/CGMSeriesLunchPat'+str(i+1)+'.csv')
        df_date_num_pat = df_date_num_pat.append(df_date_num_lunch_pat.iloc[:, 0 : 31], ignore_index = True, sort = False)
        df_glucose_pat = df_glucose_pat.append(df_series_lunch_pat.iloc[:, 0 : 31], ignore_index = True, sort = False)


    #Clean the data
    clean_data()

    #Check for rows having NaN in data:
    nan_rows = pd.isnull(df_glucose_pat).any(1).to_numpy().nonzero()[0].tolist()
    #print (nan_rows)
    # Drop these rows from Series and Date Time dataframe
    df_glucose_pat = df_glucose_pat.drop(nan_rows)
    df_date_num_pat = df_date_num_pat.drop(nan_rows)

    #Reset Dataframe Row Index
    df_glucose_pat.reset_index(drop = True, inplace = True)
    df_date_num_pat.reset_index(drop = True, inplace = True)

    #Copying the df_glucose_pat to a new DataFrame for extracting feature
    df_glucose_pat_cp = df_glucose_pat.copy()
    df_glucose_pat_cp = df_glucose_pat_cp.astype(int)

    # for i in range (len(df_glucose_pat)):
    #     plt.plot(df_date_num_pat.iloc[i],df_glucose_pat.iloc[i])
    #     plt.xlabel('Time')
    #     plt.ylabel('Glucose')
    #     plt.show()
    # Feature Extraction
    calFFT()
    calWelch()
    calCgmVelocity()
    calPolyfit()

    feature_matrix = df_glucose_pat.iloc[:,31:]

    feature_matrix = feature_matrix.astype(float)
    #print (feature_matrix)

    #Apply PCA
    new_feature_matrix = performPCA()
    print (new_feature_matrix)





