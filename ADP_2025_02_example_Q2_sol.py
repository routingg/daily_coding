#################
# Q2 
import math

def calculate_variance_and_std(data, is_sample=True):
    """
    분산과 표준편차를 계산하는 함수
    
    Args:
        data: 숫자 리스트
        is_sample: True면 표본 분산, False면 모집단 분산
    
    Returns:
        dict: {'variance': 분산, 'std_dev': 표준편차, 'mean': 평균}
    """
    if not data or len(data) == 0:
        return {'variance': 0, 'std_dev': 0, 'mean': 0}
    
    # 평균 계산
    mean = sum(data) / len(data)
    
    # 분산 계산
    squared_diffs = [(x - mean) ** 2 for x in data]
    
    if is_sample and len(data) > 1:
        # 표본 분산 (n-1로 나누기)
        variance = sum(squared_diffs) / (len(data) - 1)
    else:
        # 모집단 분산 (n으로 나누기)
        variance = sum(squared_diffs) / len(data)
    
    # 표준편차 계산
    std_dev = math.sqrt(variance)
    
    return {
        'variance': round(variance, 4),
        'std_dev': round(std_dev, 4),
        'mean': round(mean, 4)
    }

# 테스트
test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 표본 분산
sample_result = calculate_variance_and_std(test_data, is_sample=True)
print("표본 분산 결과:")
print(f"평균: {sample_result['mean']}")
print(f"분산: {sample_result['variance']}")
print(f"표준편차: {sample_result['std_dev']}")

print("\n" + "="*30)

# 모집단 분산
population_result = calculate_variance_and_std(test_data, is_sample=False)
print("모집단 분산 결과:")
print(f"평균: {population_result['mean']}")
print(f"분산: {population_result['variance']}")
print(f"표준편차: {population_result['std_dev']}")

# 추가 테스트: 성적 데이터
grades = [85, 92, 78, 96, 88, 91, 83, 89, 94, 87]
grade_stats = calculate_variance_and_std(grades, is_sample=True)
print(f"\n성적 데이터 통계:")
print(f"평균: {grade_stats['mean']}")
print(f"분산: {grade_stats['variance']}")
print(f"표준편차: {grade_stats['std_dev']}")

