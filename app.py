from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pyvi import ViTokenizer
import unicodedata
from lime.lime_text import LimeTextExplainer

# =======================================================
# 1. CẤU HÌNH & KHỞI TẠO
# =======================================================
app = Flask(__name__)

# Cấu hình thiết bị và đường dẫn
# Lưu ý: Giải nén file zip tải từ Colab và đổi tên folder thành 'model_save'
MODEL_PATH = "./model_save"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("⏳ Đang tải Model & Tokenizer (Sẽ mất khoảng 30s)...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print(f"✅ Đã tải xong Model trên thiết bị: {DEVICE}")
except Exception as e:
    print(f"❌ Lỗi tải model: {e}")
    print("Gợi ý: Kiểm tra folder 'model_save' đã được giải nén đúng chỗ chưa.")
    exit()

# Khởi tạo LIME Explainer
class_names = ["Tiêu cực", "Trung lập", "Tích cực"]

# Hàm cắt từ cho LIME: Chỉ cắt theo khoảng trắng (vì ta đã tách từ rồi)
def custom_split(text):
    return text.split()

explainer = LimeTextExplainer(
    class_names=class_names, 
    split_expression=custom_split, 
    bow=False
)

# =======================================================
# 2. BỘ XỬ LÝ TEXT (ĐỒNG BỘ VỚI TRAINING)
# =======================================================

# 2.1 Từ điển Teencode
teencode_dict = {
    "ko": "không", "k": "không", "kh": "không", "hok": "không", 
    "dc": "được", "đc": "được", "t": "tôi", "mk": "mình", 
    "thik": "thích", "iu": "yêu", "wá": "quá", "wa": "quá",
    "gíao viên": "giáo viên", "giảng viên": "giáo viên", 
    "vcl": "vô cùng", "đkm": "bực mình"
}

# 2.2 Danh sách Stopwords (Copy y hệt từ Colab)
stopwords = set([
    "thì", "là", "mà", "và", "của", "những", "cái", "việc", 
    "ở", "với", "cho", "được", "bị", "các", "có", "trong", 
    "đã", "đang", "sẽ", "cũng", "này", "kia", "đó", "ra", "vào",
    "mình", "tôi", "bạn", "nó", "họ", "rất", "quá", "lắm"
])

def clean_and_segment(text):
    """
    Quy trình xử lý: 
    Raw -> Lowercase -> Teencode -> Tách từ (PyVi) -> Xóa Stopwords
    """
    if not isinstance(text, str): return ""
    
    # Chuẩn hóa cơ bản
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    
    # Xử lý Teencode
    words = text.split()
    words = [teencode_dict.get(word, word) for word in words]
    text = ' '.join(words)
    
    # Tách từ (Sinh viên -> Sinh_viên)
    text = ViTokenizer.tokenize(text)
    
    # XÓA STOPWORDS (Bước quan trọng nhất để đồng bộ)
    tokens = text.split()
    filtered_tokens = [t for t in tokens if t not in stopwords]
    
    return ' '.join(filtered_tokens)

# =======================================================
# 3. HÀM DỰ ĐOÁN
# =======================================================
def predict_proba(texts):
    """Hàm wrapper cho LIME"""
    # Lưu ý: texts ở đây là text đã qua xử lý (clean_and_segment) từ trước
    # Nhưng ta check lại cho chắc chắn
    processed_texts = [text if "_" in text else clean_and_segment(text) for text in texts]
    
    inputs = tokenizer(
        processed_texts, 
        padding=True, 
        truncation=True, 
        max_length=100, 
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    
    return probs.cpu().numpy()

# =======================================================
# 4. API ENDPOINT
# =======================================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.json
        input_data = data.get('text', None)
        
        if not input_data:
            return jsonify({"error": "No text provided"}), 400

        if isinstance(input_data, str):
            input_list = [input_data]
        else:
            input_list = input_data

        results = []
        
        for raw_text in input_list:
            # Bước 1: Tiền xử lý (Sẽ loại bỏ từ 'bị', 'thì'...)
            processed_text = clean_and_segment(raw_text)
            
            # Nếu sau khi lọc mà chuỗi rỗng (vd user chỉ nhập "bị thì là"), xử lý ngoại lệ
            if not processed_text.strip():
                results.append({
                    "input_text": raw_text,
                    "sentiment": "Không rõ",
                    "confidence_score": "0.0%",
                    "key_factors": []
                })
                continue

            # Bước 2: Dự đoán
            probs = predict_proba([processed_text])[0]
            pred_label_idx = np.argmax(probs)
            confidence = float(probs[pred_label_idx])
            sentiment_label = class_names[pred_label_idx]
            
            # Bước 3: Giải thích (LIME)
            # Chạy trên processed_text (đã sạch bóng stopwords)
            exp = explainer.explain_instance(
                processed_text, 
                predict_proba, 
                num_features=5, 
                num_samples=1000 # 1000 mẫu để chính xác cao
            )
            
            keywords = []
            for word, weight in exp.as_list():
                if weight > 0: # Chỉ lấy từ ủng hộ quyết định
                    keywords.append({
                        "word": word,
                        "importance": round(weight, 4)
                    })

            results.append({
                "input_text": raw_text,
                "processed_text": processed_text, # Trả về để debug xem đã sạch chưa
                "sentiment": sentiment_label,
                "confidence_score": f"{confidence * 100:.2f}%",
                "key_factors": keywords
            })

        return jsonify({"status": "success", "results": results})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)