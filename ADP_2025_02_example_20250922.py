"""
통계 분석 문제 모음

【문제 1: 단일표본 t검정】
어떤 제품의 무게가 평균 10g, 분산이 5이라고 주장하는 제조업체가 있다. 
이를 검증하기 위해 100개의 표본을 추출하여 무게를 측정했다.
T-검정 통계량을 계산하시오

【문제 2: 변이계수 분석】
두 개의 서로 다른 생산 라인에서 제품의 품질을 비교하고자 한다.
각 라인의 변이계수를 구하고, 어느 라인이 더 안정적인지 판단하시오.

【문제 3: 왜도와 첨도 분석】
데이터의 분포 형태를 분석하여 다음을 판단하시오:
- 분포가 대칭적인가?
- 정규분포와 비교했을 때 뾰족한가 평평한가?
- 이러한 특성이 데이터 분석에 미치는 영향은?

【문제 4: 분위수와 이상치 탐지】
주어진 데이터에서 다음을 구하시오:
- 5수치 요약 (최솟값, Q1, 중위수, Q3, 최댓값)
- 사분위수 범위(IQR)
- IQR 방법을 사용한 이상치 탐지
- 각 분위수의 의미와 해석

【문제 5: 종합 분석】
서로 다른 분포(정규분포, 균등분포, 지수분포)에서 추출한 데이터의
기술통계량을 비교하고, 각 분포의 특성을 설명하시오.
"""

import numpy as np
import math

def generate_normal_data(mu0, variance, n, seed=1233):
    """
    평균이 mu0인 정규분포에서 난수를 생성하는 함수
    
    Parameters:
    mu0: 모집단 평균
    variance: 모집단 분산
    n: 생성할 데이터 개수
    seed: 난수 시드 (재현 가능한 결과를 위해)
    
    Returns:
    data: 생성된 정규분포 난수 배열
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 분산에서 표준편차 계산
    std = math.sqrt(variance)
    
    # 정규분포 난수 생성
    data = np.random.normal(mu0, std, n)
    
    return data

def calculate_coefficient_of_variation(data):
    """
    변이계수(Coefficient of Variation)를 계산하는 함수
    CV = (표준편차 / |평균|) × 100 (%)
    
    Parameters:
    data: 데이터 배열 (리스트 또는 numpy 배열)
    
    Returns:
    cv_percent: 변이계수 (백분율)
    mean: 평균
    std: 표준편차
    """
    # 데이터를 numpy 배열로 변환
    data = np.array(data)
    
    # 평균 계산
    mean = np.mean(data)
    
    # 표준편차 계산 (표본 표준편차, n-1로 나눔)
    std = np.std(data, ddof=1)
    
    # 변이계수 계산 (백분율로 표현)
    if mean != 0:
        cv_percent = (std / abs(mean)) * 100
    else:
        cv_percent = float('inf')  # 평균이 0인 경우
    
    return cv_percent, mean, std

def calculate_skewness(data):
    """
    왜도(Skewness)를 계산하는 함수
    왜도 = E[(X - μ)³] / σ³
    
    Parameters:
    data: 데이터 배열 (리스트 또는 numpy 배열)
    
    Returns:
    skewness: 왜도 값
    """
    # 데이터를 numpy 배열로 변환
    data = np.array(data)
    n = len(data)
    
    # 평균과 표준편차 계산
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # 표본 표준편차
    
    if std == 0:
        return 0  # 표준편차가 0이면 왜도는 0
    
    # 왜도 계산 (표본 왜도 공식)
    # 편향되지 않은 추정량: n/((n-1)(n-2)) * Σ((xi - x̄)/s)³
    standardized_data = (data - mean) / std
    skewness = (n / ((n - 1) * (n - 2))) * np.sum(standardized_data ** 3)
    
    return skewness

def calculate_kurtosis(data):
    """
    첨도(Kurtosis)를 계산하는 함수
    첨도 = E[(X - μ)⁴] / σ⁴ - 3 (초과 첨도)
    
    Parameters:
    data: 데이터 배열 (리스트 또는 numpy 배열)
    
    Returns:
    kurtosis: 첨도 값 (초과 첨도)
    """
    # 데이터를 numpy 배열로 변환
    data = np.array(data)
    n = len(data)
    
    # 평균과 표준편차 계산
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # 표본 표준편차
    
    if std == 0:
        return 0  # 표준편차가 0이면 첨도는 0
    
    # 첨도 계산 (표본 첨도 공식)
    # 편향되지 않은 추정량
    standardized_data = (data - mean) / std
    
    # 4차 모멘트 계산
    m4 = np.sum(standardized_data ** 4) / n
    
    # 편향 조정된 초과 첨도
    # Fisher's definition (excess kurtosis = kurtosis - 3)
    kurtosis = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * m4 - 3 * (n - 1))
    
    return kurtosis

def calculate_quantiles(data, quantiles=[0.25, 0.5, 0.75]):
    """
    분위수(Quantiles)를 계산하는 함수
    
    Parameters:
    data: 데이터 배열 (리스트 또는 numpy 배열)
    quantiles: 계산할 분위수 리스트 (기본값: [0.25, 0.5, 0.75])
    
    Returns:
    quantile_values: 계산된 분위수 값들의 딕셔너리
    """
    # 데이터를 numpy 배열로 변환하고 정렬
    data = np.array(data)
    sorted_data = np.sort(data)
    n = len(data)
    
    quantile_values = {}
    
    for q in quantiles:
        if q < 0 or q > 1:
            raise ValueError("분위수는 0과 1 사이의 값이어야 합니다.")
        
        # 분위수 계산 (선형 보간법 사용)
        # R-6 방법 (numpy의 기본값과 동일)
        position = q * (n - 1)
        lower_index = int(position)
        upper_index = min(lower_index + 1, n - 1)
        
        if lower_index == upper_index:
            quantile_value = sorted_data[lower_index]
        else:
            # 선형 보간
            weight = position - lower_index
            quantile_value = (1 - weight) * sorted_data[lower_index] + weight * sorted_data[upper_index]
        
        quantile_values[q] = quantile_value
    
    return quantile_values

def calculate_five_number_summary(data):
    """
    5수치 요약(Five Number Summary)을 계산하는 함수
    최솟값, Q1, 중위수, Q3, 최댓값
    
    Parameters:
    data: 데이터 배열 (리스트 또는 numpy 배열)
    
    Returns:
    summary: 5수치 요약 딕셔너리
    """
    data = np.array(data)
    
    # 기본 통계량 계산
    min_val = np.min(data)
    max_val = np.max(data)
    
    # 분위수 계산
    quantiles = calculate_quantiles(data, [0.25, 0.5, 0.75])
    
    summary = {
        'minimum': min_val,
        'Q1': quantiles[0.25],
        'median': quantiles[0.5],
        'Q3': quantiles[0.75],
        'maximum': max_val,
        'IQR': quantiles[0.75] - quantiles[0.25]  # 사분위수 범위
    }
    
    return summary

def single_sample_ttest(data, mu0):
    """
    단일표본 t검정의 t 통계량을 직접 계산하는 함수
    
    Parameters:
    data: 표본 데이터 (리스트 또는 numpy 배열)
    mu0: 검정하고자 하는 모집단 평균 (귀무가설의 평균)
    
    Returns:
    t_statistic: t 통계량
    sample_mean: 표본 평균
    sample_std: 표본 표준편차
    n: 표본 크기
    df: 자유도
    """
    
    # 데이터를 numpy 배열로 변환
    data = np.array(data)
    
    # 표본 크기
    n = len(data)
    
    # 표본 평균 계산
    sample_mean = np.sum(data) / n
    
    # 표본 표준편차 계산 (n-1로 나누는 불편추정량)
    variance = np.sum((data - sample_mean)**2) / (n - 1)
    sample_std = math.sqrt(variance)
    
    # 표준오차 계산
    standard_error = sample_std / math.sqrt(n)
    
    # t 통계량 계산: t = (표본평균 - 모집단평균) / 표준오차
    t_statistic = (sample_mean - mu0) / standard_error
    
    # 자유도
    df = n - 1
    
    return t_statistic, sample_mean, sample_std, n, df

def main():
    """
    메인 함수: 통계 분석 문제들을 순차적으로 해결
    """
    print("=" * 60)
    print("통계 분석 문제 해결 프로그램")
    print("=" * 60)
    
    print("\n【문제 1 해결】단일표본 t검정")
    print("-" * 40)
    
    # 임의 데이터 생성 (정규분포에서 추출)

    true_mean = 10     # 모평균
    true_variance = 5  # 모분산 
    sample_size = 10
    
    sample_data = generate_normal_data(true_mean, true_variance, sample_size, seed=42)
    
    print(f"데이터 생성 정보:")
    print(f"모집단 평균: {true_mean}")
    print(f"모집단 분산: {true_variance} (표준편차: {math.sqrt(true_variance)})")
    print(f"표본 크기: {sample_size}")
    print()
    
    print(f"생성된 표본 데이터:")
    print(f"데이터: {np.round(sample_data, 2)}")
    print()
    
    # 검정하고자 하는 모집단 평균 (귀무가설: μ = 10)
    hypothesized_mean = 10
    
    print(f"귀무가설 H0: μ = {hypothesized_mean}")
    print(f"대립가설 H1: μ ≠ {hypothesized_mean}")
    print()
    
    # t검정 수행
    t_stat, x_bar, s, n, df = single_sample_ttest(sample_data, hypothesized_mean)
    
    # 결과 출력
    print("계산 과정:")
    print(f"표본 크기 (n): {n}")
    print(f"표본 평균 (x̄): {x_bar:.4f}")
    print(f"표본 표준편차 (s): {s:.4f}")
    print(f"표준오차 (SE): {s/math.sqrt(n):.4f}")
    print(f"자유도 (df): {df}")
    print()
    
    print("t 통계량 계산:")
    print(f"t = (x̄ - μ0) / (s / √n)")
    print(f"t = ({x_bar:.4f} - {hypothesized_mean}) / ({s:.4f} / √{n})")
    print(f"t = {t_stat:.4f}")
    print()
    
    print("-" * 40)
    
    # 변이계수 계산
    cv, cv_mean, cv_std = calculate_coefficient_of_variation(sample_data)
    print(f"\n변이계수 (Coefficient of Variation):")
    print(f"평균: {cv_mean:.4f}")
    print(f"표준편차: {cv_std:.4f}")
    print(f"변이계수: {cv:.2f}%")
    print(f"해석: ", end="")
    if cv < 10:
        print("변동성이 낮음 (CV < 10%)")
    elif cv < 20:
        print("변동성이 보통 (10% ≤ CV < 20%)")
    else:
        print("변동성이 높음 (CV ≥ 20%)")
    
    # 왜도 계산
    skew = calculate_skewness(sample_data)
    print(f"\n왜도 (Skewness): {skew:.4f}")
    print(f"해석: ", end="")
    if abs(skew) < 0.5:
        print("거의 대칭적인 분포")
    elif abs(skew) < 1.0:
        print("약간 비대칭적인 분포")
    else:
        print("매우 비대칭적인 분포")
    
    if skew > 0:
        print("     → 오른쪽 꼬리가 긴 분포 (양의 왜도)")
    elif skew < 0:
        print("     → 왼쪽 꼬리가 긴 분포 (음의 왜도)")
    else:
        print("     → 완전히 대칭적인 분포")
    
    # 첨도 계산
    kurt = calculate_kurtosis(sample_data)
    print(f"\n첨도 (Kurtosis): {kurt:.4f}")
    print(f"해석: ", end="")
    if kurt > 0:
        print("정규분포보다 뾰족한 분포 (양의 초과첨도)")
    elif kurt < 0:
        print("정규분포보다 평평한 분포 (음의 초과첨도)")
    else:
        print("정규분포와 같은 첨도")
    
    print("\n【문제 5 해결】다양한 분포의 기술통계량 비교")
    print("-" * 40)
    
    # 정규분포
    normal_data = generate_normal_data(50, 100, 100, seed=42)
    normal_cv, _, _ = calculate_coefficient_of_variation(normal_data)
    normal_skew = calculate_skewness(normal_data)
    normal_kurt = calculate_kurtosis(normal_data)
    
    # 치우친 분포 (지수분포 근사)
    skewed_data = np.random.exponential(2, 100)
    np.random.seed(42)  # 시드 재설정
    skewed_data = np.random.exponential(2, 100)
    skewed_cv, _, _ = calculate_coefficient_of_variation(skewed_data)
    skewed_skew = calculate_skewness(skewed_data)
    skewed_kurt = calculate_kurtosis(skewed_data)
    
    print(f"\n정규분포 (μ=50, σ²=100):")
    print(f"  변이계수: {normal_cv:.2f}%")
    print(f"  왜도: {normal_skew:.4f}")
    print(f"  첨도: {normal_kurt:.4f}")
    
    print(f"\n지수분포 (λ=0.5):")
    print(f"  변이계수: {skewed_cv:.2f}%")
    print(f"  왜도: {skewed_skew:.4f}")
    print(f"  첨도: {skewed_kurt:.4f}")
    
    print("\n【문제 4 상세 해결】분위수 분석")
    print("-" * 40)
    
    # 원래 데이터의 분위수 계산
    print(f"\n원본 데이터의 분위수 분석:")
    
    # 다양한 분위수 계산
    various_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quantile_results = calculate_quantiles(sample_data, various_quantiles)
    
    print(f"데이터 개수: {len(sample_data)}")
    print(f"정렬된 데이터: {np.round(np.sort(sample_data), 2)}")
    print("\n분위수 결과:")
    
    for q, value in quantile_results.items():
        percentage = q * 100
        print(f"  {percentage:4.1f}% 분위수 (Q{percentage/25:.1f}): {value:.4f}")
    
    # 5수치 요약
    print(f"\n5수치 요약 (Five Number Summary):")
    five_num_summary = calculate_five_number_summary(sample_data)
    
    print(f"  최솟값: {five_num_summary['minimum']:.4f}")
    print(f"  Q1 (25%): {five_num_summary['Q1']:.4f}")
    print(f"  중위수 (50%): {five_num_summary['median']:.4f}")
    print(f"  Q3 (75%): {five_num_summary['Q3']:.4f}")
    print(f"  최댓값: {five_num_summary['maximum']:.4f}")
    print(f"  IQR (사분위수 범위): {five_num_summary['IQR']:.4f}")
    
    # 이상치 탐지
    print(f"\n이상치 탐지 (IQR 방법):")
    iqr = five_num_summary['IQR']
    q1 = five_num_summary['Q1']
    q3 = five_num_summary['Q3']
    
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    outliers = sample_data[(sample_data < lower_fence) | (sample_data > upper_fence)]
    
    print(f"  하한 경계: {lower_fence:.4f}")
    print(f"  상한 경계: {upper_fence:.4f}")
    print(f"  이상치 개수: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  이상치: {np.round(outliers, 4)}")
    else:
        print("  이상치 없음")
    
    # 다른 데이터셋과 분위수 비교
    print(f"\n다양한 분포의 분위수 비교:")
    
    # 균등분포 데이터
    uniform_data = np.random.uniform(0, 100, 50)
    np.random.seed(123)
    uniform_data = np.random.uniform(0, 100, 50)
    uniform_summary = calculate_five_number_summary(uniform_data)
    
    # 정규분포 데이터 (다른 모수)
    normal_data2 = generate_normal_data(50, 25, 50, seed=456)
    normal_summary2 = calculate_five_number_summary(normal_data2)
    
    print(f"\n균등분포 U(0,100) (n=50):")
    print(f"  Q1: {uniform_summary['Q1']:.2f}, 중위수: {uniform_summary['median']:.2f}, Q3: {uniform_summary['Q3']:.2f}")
    print(f"  IQR: {uniform_summary['IQR']:.2f}")
    
    print(f"\n정규분포 N(50,25) (n=50):")
    print(f"  Q1: {normal_summary2['Q1']:.2f}, 중위수: {normal_summary2['median']:.2f}, Q3: {normal_summary2['Q3']:.2f}")
    print(f"  IQR: {normal_summary2['IQR']:.2f}")
    
    print(f"\n원본 데이터 N(100,225) (n=25):")
    print(f"  Q1: {five_num_summary['Q1']:.2f}, 중위수: {five_num_summary['median']:.2f}, Q3: {five_num_summary['Q3']:.2f}")
    print(f"  IQR: {five_num_summary['IQR']:.2f}")
    
    # 문제 해답 요약
    print("\n" + "=" * 60)
    print("문제 해답 요약")
    print("=" * 60)
    
    print("\n【문제 1 답】단일표본 t검정 결과:")
    print(f"- 계산된 t 통계량: {t_stat:.4f}")
    
    
    print(f"\n【문제 2 답】변이계수 분석 결과:")
    print(f"- 원본 데이터 변이계수: {cv:.2f}%")
    print(f"- 해석: {cv:.2f}%는 ", end="")
    if cv < 10:
        print("변동성이 낮아 안정적")
    elif cv < 20:
        print("변동성이 보통 수준")
    else:
        print("변동성이 높아 불안정")
    
    print(f"\n【문제 3 답】왜도와 첨도 분석 결과:")
    print(f"- 왜도: {skew:.4f} → ", end="")
    if abs(skew) < 0.5:
        print("거의 대칭적 분포")
    elif skew > 0:
        print("오른쪽 꼬리가 긴 분포")
    else:
        print("왼쪽 꼬리가 긴 분포")
    print(f"- 첨도: {kurt:.4f} → ", end="")
    if kurt > 0:
        print("정규분포보다 뾰족한 분포")
    else:
        print("정규분포보다 평평한 분포")
    
    print(f"\n【문제 4 답】분위수와 이상치 탐지 결과:")
    print(f"- Q1: {five_num_summary['Q1']:.2f}, 중위수: {five_num_summary['median']:.2f}, Q3: {five_num_summary['Q3']:.2f}")
    print(f"- IQR: {five_num_summary['IQR']:.2f}")
    print(f"- 이상치 개수: {len(outliers)}개")
    
    print(f"\n【문제 5 답】분포별 특성 비교:")
    print(f"- 정규분포: 대칭적, 변이계수 {normal_cv:.1f}%")
    print(f"- 지수분포: 오른쪽 치우침, 변이계수 {skewed_cv:.1f}% (높은 변동성)")
    print(f"- 균등분포: 대칭적, 변이계수 약 30% (중간 변동성)")

if __name__ == "__main__":
    main()
