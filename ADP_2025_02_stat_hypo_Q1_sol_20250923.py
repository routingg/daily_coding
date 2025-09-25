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

def one_sample_t_test(data, mu0, alpha=0.05):
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    se = sample_std / math.sqrt(n)
    t_stat = (sample_mean - mu0) / se
    df = n - 1
    
    # scipy.stats를 사용한 정확한 계산 (양측검정)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    critical_value = stats.t.ppf(1 - alpha/2, df)
    reject = abs(t_stat) > critical_value
   #reject = p-value < alpha  
    return {
        'mean': sample_mean,
        'std': sample_std,
        'se': se,
        't_stat': t_stat,
        'df': df,
        'p_value': p_value,
        'critical': critical_value,
        'reject': reject
    }

data = generate_data()

result1 = one_sample_t_test(data['vitamin_c'], mu0=500)
print(f"\n표본 평균: {result1['mean']:.4f} mg")
print(f"표본 표준편차: {result1['std']:.4f} mg")
print(f"t 통계량: {result1['t_stat']:.4f}")
print(f"자유도: {result1['df']}")
print(f"p-value: {result1['p_value']:.4f}")
print(f"임계값: ±{result1['critical']:.4f}")
print(f"결론: {'귀무가설을 기각한다' if result1['reject'] else '귀무가설을 기각하지 않는다'}")



