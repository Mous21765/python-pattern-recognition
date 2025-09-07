from pattern_rec import housing_price


h=housing_price()
h.visualise_data()
h.create_dataset()
print('==PERCEPTRON==')
h.perceptron()
print('==Least Squares==')
h.least_squares()
print('==Regression==')
h.regression()