#################################################################
def KFold_Predict(x_test,nfolds=3,batch_size=128):
    #model = mnist_model(input_shape)
    model = zika_model(input_shape)
    yfull_test = []
    for num_fold in range(1,nfolds+1):
        weight_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)
            
        p_test = model.predict(x_test, batch_size = batch_size, verbose=2)
        yfull_test.append(p_test)
        
    result = np.array(yfull_test[0])
    for i in range(1, nfolds):
        result += np.array(yfull_test[i])
    result /= nfolds
    return result


#################################################################

def Predict(x_test):
	output = KFold_Predict(x_test)
	matrix = [[0 for x in range(2)] for y in range(2)] 
	for i in range(0, output.shape[0]):
	    if output[i] > 0.5:
             output[i] = 1
	    else:
             output[i] = 0
	    x_var = int(y_test[i])
	    y_var = int(output[i])
	    matrix[x_var][y_var] += 1

	return output, matrix     


