import numpy as np  # for data manipulation and calculation
import keras    # to implement neural network
import pandas as pd     # for data handling
import matplotlib.pyplot as plt     # to visualise outputs
import os   # to locate data and model pathways
from sklearn.preprocessing import RobustScaler  # to pre-process data
from keras.models import model_from_json    # to save and load models from file

def getDat():   # get data from csv files
    curdir = os.path.dirname(__file__)
    datsrc = os.path.join(curdir, 'datasets', '0cases.csv')     # get file pathway

    df = pd.read_csv(datsrc, parse_dates=['date'], dayfirst=True, index_col='date')     # use date as index, parse with
    # correct datetime format
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.columns[1], axis=1)     # drop unnecessary columns - save memory

    return df   # return the dataframe to the main function

def getDat2():  # function to get data from multiple files
    curdir = os.path.dirname(__file__)
    datfl = os.path.join(curdir, 'datasets')    # folder directory where all files are stored
    x = os.listdir(datfl)   # list all files in folder
    df = pd.DataFrame()     # to later append each file

    for i in range(3):  # go through each file in folder
        flnm = x[i]     # go through each file name
        print(flnm)     # make sure correct files are being opened
        datsrc = os.path.join(datfl, flnm)  # generate file path
        temp = pd.read_csv(datsrc, parse_dates=['date'], dayfirst=True, index_col='date')   # use date as index
        temp = temp.drop(temp.columns[0], axis=1)
        temp = temp.drop(temp.columns[0], axis=1)
        temp = temp.drop(temp.columns[0], axis=1)
        temp = temp.drop(temp.columns[1], axis=1)   # remove unnecessary columns
        df = pd.concat([df, temp], axis=1)  # concatenate to pandas dataframe - build up data with each file

    df = df.dropna()    # drop any rows where data isnt available (not all files start and end on same date)

    df['day_of_week'] = df.index.dayofweek  # data engineering - generate day of week data
    df['day_of_month'] = df.index.day   # generate day of month data

    return df   # return dataframe to main function

def feature_eng(df):    # data engineering function based on input dataframe
    df['day_of_week'] = df.index.dayofweek  # data engineering - generate day of week data
    df['day_of_month'] = df.index.day   # generate day of month data

    return df   # return dataframe to main function

def train_test(df, perc):   # split data into train and test samples
    train_size = int(len(df) * perc)    # get length of train sample based on percentage
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]    # split data

    return train, test  # return both datasets to main function

def transformer(train, test):   # transform data
    case_transformer = RobustScaler()   # using robust scaler
    case_transformer = case_transformer.fit(train[['newCasesBySpecimenDate']])  # generate scaler using train data
    train['newCasesBySpecimenDate'] = case_transformer.transform(train[['newCasesBySpecimenDate']])
    test['newCasesBySpecimenDate'] = case_transformer.transform(test[['newCasesBySpecimenDate']])   # transform data

    return train, test, case_transformer    # return scaled datasets as well as the transformer

def transformer2(train, test):  # transform function where multiple datasources exist
    case_transformer = RobustScaler()   # using robust scaler
    case_transformer = case_transformer.fit(train[['newCasesBySpecimenDate']])  # generate scaler using train data
    train['newCasesBySpecimenDate'] = case_transformer.transform(train[['newCasesBySpecimenDate']])
    test['newCasesBySpecimenDate'] = case_transformer.transform(test[['newCasesBySpecimenDate']])   # transform data

    featcol = list(train.columns[1:])   # generate list of column names other than 'newCasesBySpecimenDate' (index 0)
    featform = RobustScaler()   # generate new robust scaler form remaining data
    featform.fit(train[featcol].to_numpy())     # generate transformer for additional features

    train.loc[:, featcol] = featform.transform(train[featcol].to_numpy())   # transform train data of other features
    test.loc[:, featcol] = featform.transform(test[featcol].to_numpy())     # transform test data of other features

    return train, test, case_transformer    # return scaled data and transformer

def sequencer(x, y, tsteps):    # sequence data into time series data
    xout, yout = [], []     # to which each slice will be appended
    for i in range(len(x) - tsteps):    # total number of data will be less than initial amount of data - some data
        # needs to be used for the timesteps
        temp = x.iloc[i: (i+tsteps)]    # input data is of length 'timesteps'
        xout.append(temp)
        yout.append(y.iloc[i+tsteps])   # output data is of dimension 1

    return np.array(xout), np.array(yout)   # return input and output slices to main function

def create_model(x_train):  # model definition and creation
    model = keras.Sequential()  # using keras sequential model generator
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=64, input_shape=(x_train.shape[1], x_train.shape[2]))))
    # wrapping lstm layer with bidirectional layer
    model.add(keras.layers.Dropout(0.1))    # dropout layer to regularise outputs
    model.add(keras.layers.Dense(units=1))  # merge all outputs from previous layer into 1 final output
    # keras.utils.plot_model(model, to_file='model.png', show_shapes=True)  # save model to image file
    # mean_squared_error-V, mean_absolute_error-V, mean_absolute_percentage_error-X, mean_squared_logarithmic_error-X,
    # cosine_similarity-X
    # SGD-X, RMSprop-V, Adam-V, Adadelta-V, Adagrad-V, Adamax-V, Nadam-V, Ftrl
    model.compile(loss='mean_squared_error', optimizer='Adam')  # select parameters and compile model

    return model    # return model to main function

def train_model(model, x_train, y_train, epochs):   # train model based on input parameters
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, shuffle=True) #  model is fit

    return model    # and returned to main function

def model2file(model, flnm):    # model saved to file
    jfl = flnm+'.json'
    wfl = flnm+'.h5'    # joining input name with file extension e.g. 'First.h5'
    x = os.path.dirname(__file__)   # local file directory
    dir1 = os.path.join(*[x, 'models_weights', jfl])
    dir2 = os.path.join(*[x, 'models_weights', wfl])    # define new directory to store models and weights
    model_json = model.to_json()    # model converted to json object
    with open(dir1, 'w') as json_file:
        json_file.write(model_json)     # model saved to file
    model.save_weights(dir2)    # weights saved to file
    print('Model saved to file')

    return 0    # no return required

def file2model(flnm):   # to load model
    jfl = flnm+'.json'
    wfl = flnm+'.h5'    # joining input name with file extension e.g. 'First.h5'
    x = os.path.dirname(__file__)   # local file directory
    dir1 = os.path.join(*[x, 'models_weights', jfl])
    dir2 = os.path.join(*[x, 'models_weights', wfl])    # define directory to store models and weights
    json_file = open(dir1, 'r')
    loaded_model_json = json_file.read()    # files imported to model
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(dir2)     # weights loaded
    print('Model loaded from file')

    return loaded_model     # return the loaded model to the main function

def predictor(model, x_test, case_transformer, y_test):     # predicting function
    y_pred = model.predict(x_test)  # model predicts using test dataset
    y_test_inv = case_transformer.inverse_transform(y_test.reshape(1, -1))  # actual datapoints inverse transformed
    y_pred_inv = case_transformer.inverse_transform(y_pred)     # predicted datapoints inverse transformed

    return y_pred_inv, y_test_inv   # inverse transformed datapoints are returned to main function

def visuals(test, pred):    # plotting actual vs predicted data
    plt.plot(test.flatten(), marker='.', label='true')  # plot actual data
    plt.plot(pred.flatten(), 'r', marker='.', label='predicted')    # plot predicted data
    plt.legend()    # show legend
    axes = plt.gca()    # get axes object
    axes.set_ylim([0, (np.max(test)*1.3)])  # set limit on data
    plt.show()  # show plot

    return 0    # no return to main function required

def avg_error(test, pred):  # calculated average error
    test = np.array(test).flatten()     # make sure array is flat (i.e. (,X), not (1,X))
    pred = np.array(pred).flatten()     # make sure array is flat (i.e. (,X), not (1,X))
    error = np.mean(np.abs(100*(pred-test)/test))   # calculated mean of absolute of percentage difference

    fformat = "{:.2f}".format
    error = fformat(error)  # formate float to 2dp

    return 100-float(error)     # return accuracy to main function
