import matplotlib.pyplot as plt
import os

# Kayıt klasörü
output_dir = "gorseller"
os.makedirs(output_dir, exist_ok=True)

# Veri
total = 7032
churn = 2101
non_churn = total - churn

labels = ['Churn Riski Yüksek', 'Diğer Müşteriler']
sizes = [churn, non_churn]
colors = ['#FF6F61', '#6BAED6']

# Grafik oluştur
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, explode=(0.05, 0))
plt.title("Churn Riski Dağılımı – Tahmin Edilen Sonuç (Version 8)")
plt.axis('equal')
plt.tight_layout()

# Kayıt
file_path = os.path.join(output_dir, "churn_risk_pie_v8.png")
plt.savefig(file_path, dpi=300, bbox_inches='tight')

# Bilgilendirme
print(f"✅ Grafik {file_path} olarak kaydedildi.")
print("📂 Dosyanın bulunduğu tam yol:", os.path.abspath(file_path))
