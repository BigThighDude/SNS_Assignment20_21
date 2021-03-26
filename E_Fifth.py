import Utilities as utl # Import all common functions
import numpy as np  # Too many calls are made to numpy to call from utl.np - easier this way

def main(epochs, tsteps, modinfile, showgraph, dinad):  # Take input arguments from main.py file

    data = utl.getDat2()    # Get data using function from utility file - additional data imported
    data = utl.feature_eng(data)    # Feature engineering function to generate day of week/month data
    train, test = train_test(data, dinad, tsteps)   # train test split based on the number of days in advance
    # The train test function here is a local function, where instead of a percentage, the train data is taken
    # to be all the data but the last X datapoints, which are used in the test dataset
    train, test, tform = utl.transformer2(train, test)  # pretransform all data based on training set
    x_train, y_train = utl.sequencer(train, train.newCasesBySpecimenDate, tsteps)   # Sequence for time series training
    x_test, y_test = utl.sequencer(test, test.newCasesBySpecimenDate, tsteps)   # Sequence for time series testing

    if modinfile==0:    # If model has already been generated - input argument from main.py
        model = utl.create_model(x_train)   # Create the model
        model = utl.train_model(model, x_train, y_train, epochs)    # Train the model - the model is trained on all
        # datapoints except the last X (days in advance) datapoints
        utl.model2file(model, 'Fifth')  # Save model - pass file name to function

    model = utl.file2model('Fifth')     # Load model - pass file name to function
    pred, test = utl.predictor(model, x_test, tform, y_test)    # Use model to predict on test dataset - from util file
    # This prediction is done the normal way, just as a way to compare the rolling prediction vs X days in
    # advance prediction

    # This is the same as what is in the for function, but the first loop doesnt require replacing a value in xtemp,
    # so this is put up here to exclude that operation
    xtemp = x_test[-dinad]  # from the x_test dataset (which is X datapoints long), the first one is taken
    xtemp = xtemp[np.newaxis]   # the data is turned into 3d so that the predictor can be used (doesnt take in 2-d)
    pred2, test2, predui = predictor(model, xtemp, tform, y_test[-1])   # predict using local function which also
    # returns the non-inversed prediction
    pred3 = np.array(pred2)     # Keeps track of all the predictions - future predictions are appended to this

    for i in range(1, dinad):   # For the remaining days in advance steps
        xtemp = x_test[-dinad+i]    # Goes through each of the test datapoints
        xtemp[-1][0] = predui[0][0]     # Replaces the final datapoint with the prediction of the previous datapoint
        xtemp = xtemp[np.newaxis]   # Turned into 3d for the predictor
        pred2, test2, predui = predictor(model, xtemp, tform, y_test[-1])   # predicted using local function
        pred3 = np.append(pred3, pred2)     # Keeping track of all the predictions to compare later

    predflat = test.flatten()
    xdiff = np.abs(np.subtract(predflat, pred3))
    xfrac = np.divide(xdiff, predflat)
    accuracy = np.mean((1-xfrac)*100)   # The above 4 lines calculates the percentage accuracy of this new prediction
    # method (X days in advance) against the actual known values

    if showgraph==1:    # Input argument from main.py
        utl.visuals(test, pred)     # Graph the predicted vs actual data

    fformat = "{:.2f}".format
    accuracy = fformat(accuracy)    # Reformat float to 2dp

    return accuracy     # return accuracy value to main.py

def train_test(df, dinad, tsteps):  # loca train test split based on indice rather than percentage
    train_size = int(len(df)-tsteps-dinad)  # length of training dataset
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]    # split data using assignment

    return train, test  # return to main function

def predictor(model, x_test, case_transformer, y_test): # local predictor function also returns non inverted prediction
    y_pred = model.predict(x_test)  # models predicts value
    y_test_inv = case_transformer.inverse_transform(y_test.reshape(1, -1))  # inverse transform the actual values
    y_pred_inv = case_transformer.inverse_transform(y_pred)     # inverse transform the predicted values

    return y_pred_inv, y_test_inv, y_pred   # return values to main function
