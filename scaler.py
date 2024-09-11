import numpy as np
def minmax(input_array, axis_param = None):
    '''
    returns scaled version of the input array
    highest value is assigned 1
    lowest value is assigned 0

    axis_param = None , considers entire array to be a single array and 
                    scales accordingly to global maxima and global minima
    axis_param = 0 , returns values scaled column wise
    axis_param = 1, returns values scaled row wise
    '''
    min_val = np.min(input_array, axis = axis_param)
    max_val = np.max(input_array, axis = axis_param)
    
    minmax_scaled_array = (input_array - min_val) / (max_val - min_val)
    return minmax_scaled_array

def normalize(input_array, axis_param = None):
    '''
    returns normalized version of the input array
    in terms of the z-score

    axis_param = None , considers entire array to be a single array and 
                    scales accordingly to global mean and global std deviation
    axis_param = 0 , returns values scaled column wise
    axis_param = 1, returns values scaled row wise
    '''
    mean_val = np.mean(array, axis = axis_param)
    std_deviation_val = np.std(array, axis = axis_param)

    normalized_array = ( input_array - mean_val ) / std_deviation_val
    return normalized_array

array = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print(normalize(array))