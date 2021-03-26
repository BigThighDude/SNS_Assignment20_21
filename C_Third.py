import Utilities as utl # Import all common functions

def main(perc, epochs, tsteps, modinfile, showgraph):   # Take input arguments from main.py file

    data = utl.getDat() # Get data using function from utility file
    data = utl.feature_eng(data)    # Feature engineering function to generate day of week/month data
    train, test = utl.train_test(data, perc)    # train test split based on percentage input argument
    train, test, tform = utl.transformer(train, test)   # pretransform all data based on training set
    x_train, y_train = utl.sequencer(train, train.newCasesBySpecimenDate, tsteps)   # Sequence for time series training
    x_test, y_test = utl.sequencer(test, test.newCasesBySpecimenDate, tsteps)   # Sequence for time series testing

    if modinfile==0:    # If model has already been generated - input argument from main.py
        model = utl.create_model(x_train)   # Create the model
        model = utl.train_model(model, x_train, y_train, epochs)    # Train the model
        utl.model2file(model, 'Third')  # Save model - pass file name to function

    model = utl.file2model('Third') # Load model - pass file name to function
    pred, test = utl.predictor(model, x_test, tform, y_test)    # Use model to predict on test dataset - from util file

    if showgraph==1:    # Input argument from main.py
        utl.visuals(test, pred)     # Graph the predicted vs actual data

    error = utl.avg_error(test, pred)   # Calculate average error over test set

    return error    # return error value to main.py
