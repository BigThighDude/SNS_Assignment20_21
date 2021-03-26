import A_First as fst   # First model - raw data
import B_Second as scd  # Second model - data scaling
import C_Third as thd   # Third model - data scaling and feature engineering
import D_Fourth as frt  # Fourth model - data scaling, feature engineering, additional data sources
import E_Fifth as ffh   # Fifth model - model same as fourth (trained separately), used to predict X days in advance
import time             # Keep track of how long the execution time is

pc = 0.9    # train test split
epc = 500   # number of epochs
tsp = 5     # number of time steps
mif = 0     # model already generated? (model in file?) - 0/no, 1/yes
sg = 1      # show graph - 0/no, 1/yes
dia = 7     # days in advance

start = time.time()
e1 = fst.main(perc=pc, epochs=epc, tsteps=tsp, modinfile=mif, showgraph=sg)     # First model
e2 = scd.main(perc=pc, epochs=epc, tsteps=tsp, modinfile=mif, showgraph=sg)     # Second model
e3 = thd.main(perc=pc, epochs=epc, tsteps=tsp, modinfile=mif, showgraph=sg)     # Third model
e4 = frt.main(perc=pc, epochs=epc, tsteps=tsp, modinfile=mif, showgraph=sg)     # Fourth model
e5 = ffh.main(epochs=epc, tsteps=tsp, modinfile=mif, showgraph=sg, dinad=dia)   # Fifth model
end = time.time()

e = [e1, e2, e3, e4, e5]    # Average percentage difference between predicted and actual values

for i in range(len(e)):
    print('Accuracy of model ', i+1, ':\t', e[i], '%')  # For each model
print('Time taken:\t', end-start)
