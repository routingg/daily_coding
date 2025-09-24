#%pip install scipy

import numpy as np
import math
from scipy import stats

def generate_data():
    np.random.seed(42)
    
    # Problem 1 data
    vitamin_c = np.random.normal(495, 15, 20)
    
    # Problem 2 data
    before_weight = np.random.normal(70, 10, 15)
    weight_loss = np.random.normal(2, 1.5, 15)
    after_weight = before_weight - weight_loss
    
    # Problem 3 data
    method_a = np.random.normal(75, 8, 12)
    method_b = np.random.normal(80, 9, 15)
    
    # Problem 4 data
    fertilizer_a = np.random.normal(25, 3, 10)
    fertilizer_b = np.random.normal(28, 3.5, 12)
    fertilizer_c = np.random.normal(26, 2.8, 11)
    
    return {
        'vitamin_c': vitamin_c,
        'before_weight': before_weight,
        'after_weight': after_weight,
        'method_a': method_a,
        'method_b': method_b,
        'fertilizer_a': fertilizer_a,
        'fertilizer_b': fertilizer_b,
        'fertilizer_c': fertilizer_c
    }
