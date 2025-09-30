# -*- coding: utf-8 -*-
"""
Statistical Hypothesis Testing Problems
"""

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

def paired_t_test(before, after, alpha=0.05):
    diff = before - after
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se = std_diff / math.sqrt(n)
    t_stat = mean_diff / se
    df = n - 1
    
    # scipy.stats를 사용한 정확한 계산 (단측검정: 우측)
    p_value = 1 - stats.t.cdf(t_stat, df)
    critical_value = stats.t.ppf(1 - alpha, df)
    reject = t_stat > critical_value
    
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'se': se,
        't_stat': t_stat,
        'df': df,
        'p_value': p_value,
        'critical': critical_value,
        'reject': reject
    }

def independent_t_test(group1, group2, alpha=0.05):
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled variance
    pooled_var = ((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2)
    pooled_std = math.sqrt(pooled_var)
    se = pooled_std * math.sqrt(1/n1 + 1/n2)
    
    t_stat = (mean1 - mean2) / se
    df = n1 + n2 - 2
    
    # scipy.stats를 사용한 정확한 계산 (양측검정)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    critical_value = stats.t.ppf(1 - alpha/2, df)
    reject = abs(t_stat) > critical_value
    #reject = p_value < alpha  # H0 reject
    return {
        'mean1': mean1,
        'mean2': mean2,
        'pooled_std': pooled_std,
        'se': se,
        't_stat': t_stat,
        'df': df,
        'p_value': p_value,
        'critical': critical_value,
        'reject': reject
    }

def one_way_anova(group1, group2, group3, alpha=0.05):
    groups = [group1, group2, group3]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    #n_total = len(group1) + len(group2) + len(group3)
    
    group_means = [np.mean(g) for g in groups]
    
    # 구현해보세요 온라인강의 과제  
  
    
    # scipy.stats를 사용한 정확한 계산
    p_value = 1 - stats.f.cdf(f_stat, df_between, df_within)
    critical_value = stats.f.ppf(1 - alpha, df_between, df_within)
    reject = f_stat > critical_value
    
    return {
        'group_means': group_means,
        'overall_mean': overall_mean,
        'ssb': ssb,
        'ssw': ssw,
        'df_between': df_between,
        'df_within': df_within,
        'msb': msb,
        'msw': msw,
        'f_stat': f_stat,
        'p_value': p_value,
        'critical': critical_value,
        'reject': reject
    }

def solve_problems():

    data = generate_data()
    
    print("\n" + "=" * 80)
    print("문제 해결 및 결과")
    print("=" * 80)
    
    # 문제 1
    print("\n【문제 1 해결: 일표본 t검정】")
    print("-" * 50)
    print("H₀: μ = 500 (제약회사의 주장이 맞다)")
    print("H₁: μ ≠ 500 (제약회사의 주장이 틀리다)")
    print(f"생성된 데이터: {np.round(data['vitamin_c'], 2)}")
    
    result1 = one_sample_t_test(data['vitamin_c'], 500)
    print(f"\n표본 평균: {result1['mean']:.4f} mg")
    print(f"표본 표준편차: {result1['std']:.4f} mg")
    print(f"t 통계량: {result1['t_stat']:.4f}")
    print(f"자유도: {result1['df']}")
    print(f"p-value: {result1['p_value']:.4f}")
    print(f"임계값: ±{result1['critical']:.4f}")
    print(f"결론: {'귀무가설을 기각한다' if result1['reject'] else '귀무가설을 기각하지 않는다'}")
    
    # 문제 2
    print("\n【문제 2 해결: 대응표본 t검정】")
    print("-" * 50)
    print("H₀: μd ≤ 0 (다이어트 효과가 없다)")
    print("H₁: μd > 0 (다이어트 효과가 있다)")
    print(f"다이어트 전 체중: {np.round(data['before_weight'], 1)}")
    print(f"다이어트 후 체중: {np.round(data['after_weight'], 1)}")
    
    result2 = paired_t_test(data['before_weight'], data['after_weight'])
    print(f"\n체중 감소량 평균: {result2['mean_diff']:.4f} kg")
    print(f"t 통계량: {result2['t_stat']:.4f}")
    print(f"자유도: {result2['df']}")
    print(f"p-value: {result2['p_value']:.4f}")
    print(f"임계값: {result2['critical']:.4f}")
    print(f"결론: {'귀무가설을 기각한다' if result2['reject'] else '귀무가설을 기각하지 않는다'}")
    
    # 문제 3
    print("\n【문제 3 해결: 독립표본 t검정】")
    print("-" * 50)
    print("H₀: μA = μB (두 교육방법의 평균이 같다)")
    print("H₁: μA ≠ μB (두 교육방법의 평균이 다르다)")
    print(f"A방법 점수: {np.round(data['method_a'], 1)}")
    print(f"B방법 점수: {np.round(data['method_b'], 1)}")
    
    result3 = independent_t_test(data['method_a'], data['method_b'])
    print(f"\nA방법 평균: {result3['mean1']:.4f}")
    print(f"B방법 평균: {result3['mean2']:.4f}")
    print(f"합동표준편차: {result3['pooled_std']:.4f}")
    print(f"t 통계량: {result3['t_stat']:.4f}")
    print(f"자유도: {result3['df']}")
    print(f"p-value: {result3['p_value']:.4f}")
    print(f"임계값: ±{result3['critical']:.4f}")
    print(f"결론: {'귀무가설을 기각한다' if result3['reject'] else '귀무가설을 기각하지 않는다'}")
    
    # 문제 4
    # 출력도 구현하세요
    
if __name__ == "__main__":
    solve_problems()
