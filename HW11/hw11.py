

# eq 49
def loss_function(test_data, train_data):
    return pow((train_data-test_data),2)

# eq 50
def emp_risk_function(test_dataset, train_dataset):
    res = 0
    for i in range (0, len(test_dataset)):
        res += loss_function(test_dataset[i], train_dataset[i])
    return res/(len(test_dataset)+1)

step_size = 0.001



#data, data, label 
input_data = [[1, 1, 0], [0, 0, 0], [0, 1, 1], [1, 0, 1]]

# MLP Structure
input_layer = [x, y, 1]

hidden_layer = [a, b, 1]

