from termcolor import colored
import numpy as np

def sigmoid_test(sigmoid):
    if sigmoid(np.array([0,2]))[0] == 0.5 and sigmoid(np.array([0,2]))[1] == 0.8807970779778823:
        print(colored('Selamat! Jawaban Anda benar', 'green'))
    else:
        print(colored('Jawaban Anda masih salah', 'red'))
		
def initialize_with_zeros_test(initialize_with_zeros):
	w, b = initialize_with_zeros(2)
	if str(w.shape) == '(2, 1)' and w[0] == np.array([0.]) and w[1] == np.array([0.]) and b == 0.0:
		print(colored('Selamat! Jawaban Anda benar', 'green'))
	else:
		print(colored('Jawaban Anda masih salah', 'red'))
		
def propagate_test(propagate):
	w =  np.array([[1.], [2]])
	b = 1.5
	X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
	Y = np.array([[1, 1, 0]])
	grads, cost = propagate(w, b, X, Y)
	
	if grads["dw"][0][0] == 0.2507153166184082 and grads["dw"][1][0] == -0.06604096325829123 and grads["db"] == -0.1250040450043965:
		print(colored('Selamat! Jawaban Anda benar', 'green'))
	else:
		print(colored('Jawaban Anda masih salah', 'red'))
		
def optimize_test(optimize):
    w =  np.array([[1.], [2]])
    b = 1.5
    X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
    Y = np.array([[1, 1, 0]])
    
    params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
    
    check_params = "{'w': array([[0.80956046],\n       [2.0508202 ]]), 'b': 1.5948713189708588}"
    check_grads = "{'dw': array([[ 0.17860505],\n       [-0.04840656]]), 'db': -0.08888460336847771}"
    check_costs = "[array(0.15900538)]"
        
    if str(params) == check_params and str(grads) == check_grads and str(costs) == check_costs:
        print(colored('Selamat! Jawaban Anda benar', 'green'))
    else:
        print(colored('Jawaban Anda masih salah', 'red'))        

def predict_test(predict):
    w = np.array([[0.1124579], [0.23106775]])
    b = -0.3
    X = np.array([[1., -1.1, -3.2],[1.2, 2., 0.1]])
    
    check_predict = '[[1. 1. 0.]]'
        
    if str(predict(w, b, X)) == check_predict :
        print(colored('Selamat! Jawaban Anda benar', 'green'))
    else:
        print(colored('Jawaban Anda masih salah', 'red'))        
		
def model_test(model):
	check_model = '<function'
	if str(model)[0:9] == check_model:
		print(colored('Selamat! Jawaban Anda benar', 'green'))
	else:
		print(colored('Jawaban Anda masih salah', 'red'))   		