import Utilities as utl # Import all common functions

def main(perc, epochs, tsteps, modinfile, showgraph):   # Take input arguments from main.py file

    data = utl.getDat()     # Get data using function from utility file
    train, test = utl.train_test(data, perc)    # train test split based on percentage input argument
    x_train, y_train = utl.sequencer(train, train.newCasesBySpecimenDate, tsteps)   # Sequence for time series training
    x_test, y_test = utl.sequencer(test, test.newCasesBySpecimenDate, tsteps)   # Sequence for time series testing

    if modinfile==0:    # If model has already been generated - input argument from main.py
        model = utl.create_model(x_train)   # Create the model
        model = utl.train_model(model, x_train, y_train, epochs)    # Train the model
        utl.model2file(model, 'First')  # Save model - pass file name to function

    model = utl.file2model('First')     # Load model - pass file name to function
    pred = predictor(model, x_test)     # Use model to predict on test dataset - NOT from util file

    if showgraph==1:    # Input argument from main.py
        utl.visuals(y_test, pred)   # Graph the predicted vs actual data

    error = utl.avg_error(y_test, pred)     # Calculate average error over test set
    fformat = "{:.2f}".format
    error = fformat(error)  # Reformat float to 2dp

    return error    # return error value to main.py

def predictor(model, x_test):   # predictor model which doesnt inverse transform (as the utility file does)
    y_pred = model.predict(x_test)  # simple prediction and return

    return y_pred
