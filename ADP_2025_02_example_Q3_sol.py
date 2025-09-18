##### 
# Q3 
import math

def calculate_sample_correlation(x_data, y_data):
    """
    표본 상관계수를 계산하는 함수
    
    Args:
        x_data: 첫 번째 변수의 데이터 리스트
        y_data: 두 번째 변수의 데이터 리스트
    
    Returns:
        dict: {'correlation': 상관계수, 'n': 표본크기, 'covariance': 공분산, 'x_std': x 표준편차, 'y_std': y 표준편차}
    """
    n_x = len(x_data)
    n_y = len(y_data)
    
    # 데이터 길이가 다른 경우
    if n_x != n_y:
        return {
            'correlation': 0,
            'n': 0,
            'covariance': 0,
            'x_std': 0,
            'y_std': 0,
            'error': f'데이터 길이가 다릅니다. x: {n_x}, y: {n_y}'
        }
    
    # 표본 크기가 2 미만인 경우
    if n_x < 2:
        return {
            'correlation': 0,
            'n': n_x,
            'covariance': 0,
            'x_std': 0,
            'y_std': 0,
            'error': '표본 크기가 2 미만입니다. 상관계수를 계산할 수 없습니다.'
        }
    
    # 평균 계산
    x_mean = sum(x_data) / n_x
    y_mean = sum(y_data) / n_y
    
    # 편차 계산
    x_deviations = [x - x_mean for x in x_data]
    y_deviations = [y - y_mean for y in y_data]
    
    # 편차의 제곱합 계산
    x_squared_deviations = [d**2 for d in x_deviations]
    y_squared_deviations = [d**2 for d in y_deviations]
    
    sum_x_squared = sum(x_squared_deviations)
    sum_y_squared = sum(y_squared_deviations)
    
    # 표준편차 계산 (표본 표준편차)
    x_std = math.sqrt(sum_x_squared / (n_x - 1))
    y_std = math.sqrt(sum_y_squared / (n_y - 1))
    
    # 공분산 계산 (표본 공분산)
    cross_products = [x_deviations[i] * y_deviations[i] for i in range(n_x)]
    covariance = sum(cross_products) / (n_x - 1)
    
    # 상관계수 계산
    if x_std == 0 or y_std == 0:
        correlation = 0
    else:
        correlation = covariance / (x_std * y_std)
    
    return {
        'correlation': round(correlation, 4),
        'n': n_x,
        'covariance': round(covariance, 4),
        'x_std': round(x_std, 4),
        'y_std': round(y_std, 4)
    }

# 테스트 1: 양의 상관관계
print("=== 테스트 1: 양의 상관관계 ===")
x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y1 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
result1 = calculate_sample_correlation(x1, y1)
print(f"x 데이터: {x1}")
print(f"y 데이터: {y1}")
print(f"상관계수: {result1['correlation']}")
print(f"표본크기: {result1['n']}")
print(f"공분산: {result1['covariance']}")
print(f"x 표준편차: {result1['x_std']}")
print(f"y 표준편차: {result1['y_std']}")

print("\n" + "="*50)
