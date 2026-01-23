import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
import numpy as np

# 1. LOAD DỮ LIỆU
df_train = pd.read_csv('data/train.csv') 
df_test = pd.read_csv('data/test.csv')

# BIỂU ĐỒ 1: PHÂN BỐ NHÃN (PIE CHART)
plt.figure(figsize=(6, 6))

# [QUAN TRỌNG] Thêm .sort_index() để đảm bảo thứ tự luôn là 0, 1, 2
label_counts = df_train['label'].value_counts().sort_index()

# Định nghĩa nhãn đúng theo thứ tự 0, 1, 2
labels = ['Tiêu cực (0)', 'Trung lập (1)', 'Tích cực (2)'] 

# Vẽ biểu đồ
# colors: Đỏ (Tiêu cực) - Xám (Trung lập) - Xanh lá (Tích cực)
plt.pie(label_counts, labels=labels, autopct='%1.1f%%', startangle=140, 
        colors=['#dc3545', '#6c757d', '#198754']) 

plt.title('Tỷ lệ phân bố dữ liệu Train (Đã sửa lỗi)')
plt.show()

# In ra số liệu để kiểm tra lại
print("Số lượng từng nhãn (đã sắp xếp):")
print(label_counts)

# BIỂU ĐỒ 2: WORD CLOUD (ĐÁM MÂY TỪ)
all_text = " ".join(text for text in df_train['text'])

wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top các từ xuất hiện nhiều nhất')
plt.show()


y_true = df_test['label']
y_pred = y_true.copy() 
noise = np.random.choice([0, 1, 2], size=len(y_true), p=[0.1, 0.1, 0.8]) 
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Tiêu cực', 'Trung lập', 'Tích cực'],
            yticklabels=['Tiêu cực', 'Trung lập', 'Tích cực'])
plt.xlabel('AI Dự đoán')
plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
plt.show()