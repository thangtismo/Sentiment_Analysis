import pandas as pd
from datasets import load_dataset
from pyvi import ViTokenizer
import re
import unicodedata
import os

# 1. CẤU HÌNH TỪ ĐIỂN TEENCODE & EMOJI
# Đây là danh sách các từ viết tắt sinh viên hay dùng cần quy đổi về chuẩn
teencode_dict = {
    "ko": "không", "k": "không", "kh": "không", "hok": "không", 
    "dc": "được", "đc": "được", "dk": "được",
    "t": "tôi", "mk": "mình", "mn": "mọi người",
    "thik": "thích", "thix": "thích", "iu": "yêu",
    "ck": "chồng", "vk": "vợ",
    "nt": "nhắn tin", "b": "bạn",
    "trc": "trước", "h": "giờ",
    "wá": "quá", "wa": "quá", "qá": "quá",
    "gíao viên": "giáo viên", "giảng viên": "giáo viên", 
    "hoc": "học", "hs": "học sinh", "sv": "sinh viên",
    "vcl": "vô cùng", "đkm": "bực mình", "v~": "vậy",
    "bt": "bình thường", "bth": "bình thường"
}

# 2. CÁC HÀM XỬ LÝ TEXT
def normalize_text(text):
    """Chuẩn hóa Unicode về dạng dựng sẵn (NFC)"""
    return unicodedata.normalize('NFC', text)

def remove_noise(text):
    """Xóa các ký tự không cần thiết, giữ lại dấu câu cơ bản"""
    text = text.lower() # Chuyển về chữ thường
    text = re.sub(r'<.*?>', '', text) # Xóa HTML tags
    # Xóa ký tự đặc biệt nhưng giữ lại dấu tiếng Việt và dấu câu quan trọng
    text = re.sub(r'[^\w\s\d.,!?áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]', ' ', text)
    return text

def replace_teencode(text):
    """Thay thế teencode bằng từ chuẩn"""
    words = text.split()
    processed_words = [teencode_dict.get(word, word) for word in words]
    return ' '.join(processed_words)

def segment_text(text):
    """Tách từ tiếng Việt bằng PyVi (Thêm dấu gạch dưới nối từ ghép)"""
    # PyVi sẽ biến "sinh viên" -> "sinh_viên"
    return ViTokenizer.tokenize(text)

def preprocess_pipeline(text):
    """Gộp tất cả các bước trên thành 1 pipeline"""
    if not isinstance(text, str): return ""
    text = normalize_text(text)
    text = remove_noise(text)
    text = replace_teencode(text)
    text = segment_text(text)
    return text.strip()

# 3. HÀM MAIN THỰC THI
def main():
    print("⏳ Đang tải dataset 'uitnlp/vietnamese_students_feedback' từ Hugging Face...")
    # Load dataset (đã có sẵn split train/test trên Hugging Face)
    dataset = load_dataset("uitnlp/vietnamese_students_feedback")
    
    # Chuyển sang Pandas DataFrame để dễ xử lý
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    
    print(f"✅ Đã tải xong. Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Áp dụng tiền xử lý
    print("⏳ Đang tiến hành tiền xử lý (Clean & Segment)... Vui lòng đợi...")
    
    # Cột chứa text là 'sentence', cột nhãn là 'sentiment'
    train_df['sentence_processed'] = train_df['sentence'].apply(preprocess_pipeline)
    test_df['sentence_processed'] = test_df['sentence'].apply(preprocess_pipeline)
    
    # Chọn lại cột cần thiết để lưu
    # Label mapping: 0 (Tiêu cực), 1 (Trung lập), 2 (Tích cực) - Giữ nguyên số để train
    final_train = train_df[['sentence_processed', 'sentiment']]
    final_test = test_df[['sentence_processed', 'sentiment']]
    
    # Đổi tên cột cho chuẩn format của model sau này
    final_train.columns = ['text', 'label']
    final_test.columns = ['text', 'label']
    
    # Tạo thư mục data nếu chưa có
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Lưu file CSV
    final_train.to_csv('data/train.csv', index=False)
    final_test.to_csv('data/test.csv', index=False)
    
    print("\n🎉 XỬ LÝ HOÀN TẤT!")
    print("📁 Dữ liệu đã lưu tại:")
    print("   - data/train.csv")
    print("   - data/test.csv")
    
    # In thử vài dòng để kiểm tra
    print("\n🔍 Preview dữ liệu đã xử lý (5 dòng đầu train):")
    print(final_train.head())

if __name__ == "__main__":
    main()