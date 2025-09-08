from scipy.stats import pearsonr,spearmanr,kendalltau
import numpy as np
from sklearn.metrics import mean_squared_error



# Pearson	Kendall	RMSE


import numpy as np

def evaluate(GT,Pre):
    GT = np.array(GT)
    Pre = np.array(Pre)
    
    Pearson_value,p_value = pearsonr(GT,Pre)
    # print('pearsonr: r_row={}, p_value={}'.format(Pearson_value, p_value))
    
    Kendall_value,p_value = kendalltau(GT,Pre)
    # print('kendalltau: r_row={}, p_value={}'.format(Kendall_value, p_value))
    
    mse = mean_squared_error(GT,Pre)
    rmse=np.sqrt(mse)

    
    return Pearson_value, Kendall_value, rmse
 
