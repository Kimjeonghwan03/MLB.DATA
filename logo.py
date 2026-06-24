import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 3))

bg_color = '#002D72'
ax.set_facecolor(bg_color)
fig.patch.set_facecolor(bg_color)

ax.axis('off')

plt.text(0.5, 0.65, 'MLB Data Sentiment Analysis',
         fontsize=35, fontweight='bold', color='white',
         ha='center', va='center')

plt.text(0.5, 0.35, 'Powered by MobileBERT | MLB.com vs Reddit',
         fontsize=18, fontweight='bold', color='#FF4500',
         ha='center', va='center')

plt.savefig('mlb_project_logo.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print("✅ 멋진 배너 이미지 [mlb_project_logo.png] 파일이 성공적으로 생성되었습니다!")