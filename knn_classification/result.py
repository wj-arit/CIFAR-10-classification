import matplotlib.pyplot as plt

# 데이터 예시 (네 실제 결과로 교체 가능)
k_values = [3, 5, 7, 15]
accuracies = [0.3305, 0.3398, 0.3358, 0.3410]

plt.figure(figsize=(7, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8, color='royalblue')

# 각 점에 K값과 정확도 표시
for x, y in zip(k_values, accuracies):
    plt.text(x, y + 0.0008, f"K={x}\n{y:.3f}",
             ha='center', va='bottom', fontsize=10, color='black')

# 제목과 축 레이블
plt.title("KNN Accuracy according to K Value", fontsize=14, weight='bold')
plt.xlabel("K Value", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)

# 격자와 시각 효과
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(min(accuracies) - 0.002, max(accuracies) + 0.003)
plt.tight_layout()
plt.show()
