# Q5 
import math

def calculate_two_sample_t_statistic(sample1, sample2):
    """
    등분산 가정 하에서 독립 2표본 t-검정을 위한 T 통계량을 계산하는 함수
    
    Args:
        sample1: 첫 번째 표본 데이터 리스트
        sample2: 두 번째 표본 데이터 리스트
    
    Returns:
        dict: {'t_statistic': t 통계량, 'sample1_mean': 표본1 평균, 'sample2_mean': 표본2 평균, 
               'sample1_std': 표본1 표준편차, 'sample2_std': 표본2 표준편차, 
               'pooled_variance': 합동분산, 'df': 자유도}
    """
    n1 = len(sample1)
    n2 = len(sample2)
    
    # 표본 크기가 2 미만인 경우
    if n1 < 2 or n2 < 2:
        return {
            't_statistic': 0,
            'sample1_mean': sum(sample1) / n1 if n1 > 0 else 0,
            'sample2_mean': sum(sample2) / n2 if n2 > 0 else 0,
            'sample1_std': 0,
            'sample2_std': 0,
            'pooled_variance': 0,
            'df': 0,
            'error': '각 표본의 크기가 2 미만입니다. 2표본 t-검정을 수행할 수 없습니다.'
        }
    
    # 각 표본의 평균 계산
    mean1 = sum(sample1) / n1
    mean2 = sum(sample2) / n2
    
    # 각 표본의 분산 계산 (표본 분산)
    squared_deviations1 = [(x - mean1) ** 2 for x in sample1]
    squared_deviations2 = [(y - mean2) ** 2 for y in sample2]
    
    var1 = sum(squared_deviations1) / (n1 - 1)
    var2 = sum(squared_deviations2) / (n2 - 1)
    
    # 각 표본의 표준편차 계산
    std1 = math.sqrt(var1)
    std2 = math.sqrt(var2)
    
    # 합동분산 계산: sₚ² = [(n₁-1)s₁² + (n₂-1)s₂²] / (n₁ + n₂ - 2)
    pooled_variance = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    
    # 평균 차이
    mean_diff = mean1 - mean2
    
    # 표준오차 계산: √(sₚ²(1/n₁ + 1/n₂))
    standard_error = math.sqrt(pooled_variance * (1/n1 + 1/n2))
    
    # T 통계량 계산: t = (x̄₁ - x̄₂) / √(sₚ²(1/n₁ + 1/n₂))
    if standard_error == 0:
        t_statistic = 0
    else:
        t_statistic = mean_diff / standard_error
    
    # 자유도: df = n₁ + n₂ - 2
    df = n1 + n2 - 2
    
    return {
        't_statistic': round(t_statistic, 4),
        'sample1_mean': round(mean1, 4),
        'sample2_mean': round(mean2, 4),
        'sample1_std': round(std1, 4),
        'sample2_std': round(std2, 4),
        'pooled_variance': round(pooled_variance, 4),
        'df': df
    }

# 테스트 1: 기본 예시
print("=== 테스트 1: 기본 예시 ===")
group1 = [85, 92, 78, 96, 88]
group2 = [91, 83, 89, 94, 87, 90]
result1 = calculate_two_sample_t_statistic(group1, group2)
print(f"그룹1 데이터: {group1}")
print(f"그룹2 데이터: {group2}")
print(f"그룹1 크기: {len(group1)}, 그룹2 크기: {len(group2)}")
print(f"그룹1 평균: {result1['sample1_mean']}")
print(f"그룹2 평균: {result1['sample2_mean']}")
print(f"그룹1 표준편차: {result1['sample1_std']}")
print(f"그룹2 표준편차: {result1['sample2_std']}")
print(f"합동분산: {result1['pooled_variance']}")
print(f"평균 차이: {result1['sample1_mean'] - result1['sample2_mean']}")
print(f"T 통계량: {result1['t_statistic']}")
print(f"자유도: {result1['df']}")

print("\n" + "="*60)

# 테스트 2: 서로 다른 크기의 표본
print("=== 테스트 2: 서로 다른 크기의 표본 ===")
large_group = [70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
small_group = [60, 65, 70]
result2 = calculate_two_sample_t_statistic(large_group, small_group)
print(f"큰 그룹 데이터: {large_group}")
print(f"작은 그룹 데이터: {small_group}")
print(f"큰 그룹 크기: {len(large_group)}, 작은 그룹 크기: {len(small_group)}")
print(f"큰 그룹 평균: {result2['sample1_mean']}")
print(f"작은 그룹 평균: {result2['sample2_mean']}")
print(f"큰 그룹 표준편차: {result2['sample1_std']}")
print(f"작은 그룹 표준편차: {result2['sample2_std']}")
print(f"합동분산: {result2['pooled_variance']}")
print(f"평균 차이: {result2['sample1_mean'] - result2['sample2_mean']}")
print(f"T 통계량: {result2['t_statistic']}")
print(f"자유도: {result2['df']}")

print("\n" + "="*60)

# 테스트 3: 동일한 크기의 표본
print("=== 테스트 3: 동일한 크기의 표본 ===")
group_a = [10, 12, 14, 16, 18]
group_b = [8, 10, 12, 14, 16]
result3 = calculate_two_sample_t_statistic(group_a, group_b)
print(f"그룹A 데이터: {group_a}")
print(f"그룹B 데이터: {group_b}")
print(f"그룹A 크기: {len(group_a)}, 그룹B 크기: {len(group_b)}")
print(f"그룹A 평균: {result3['sample1_mean']}")
print(f"그룹B 평균: {result3['sample2_mean']}")
print(f"그룹A 표준편차: {result3['sample1_std']}")
print(f"그룹B 표준편차: {result3['sample2_std']}")
print(f"합동분산: {result3['pooled_variance']}")
print(f"평균 차이: {result3['sample1_mean'] - result3['sample2_mean']}")
print(f"T 통계량: {result3['t_statistic']}")
print(f"자유도: {result3['df']}")

print("\n" + "="*60)

# 테스트 4: 예외 처리
print("=== 테스트 4: 예외 처리 ===")
# 표본 크기 부족
result4 = calculate_two_sample_t_statistic([85], [90])
print("표본 크기 부족:")
for key, value in result4.items():
    print(f"{key}: {value}")

print("\n" + "="*60)
