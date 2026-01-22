import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# ==========================================
# ⚡ CẤU HÌNH TỐI ƯU CHO LAPTOP 16GB RAM
# ==========================================
MODEL_NAME = "vinai/phobert-base"
DATA_DIR = "./data"
OUTPUT_DIR = "./model_save"

# Tối ưu bộ nhớ & tốc độ:
MAX_LEN = 100           # Giảm độ dài câu (đủ cho feedback) -> Nhanh hơn
EPOCHS = 3              # Số vòng học
BATCH_SIZE = 4          # [QUAN TRỌNG] Giảm nhỏ để vừa VRAM 4GB của Laptop
GRAD_ACCUMULATION = 4   # Tích lũy 4 bước -> Tương đương Batch size thực tế là 16
NUM_WORKERS = 2         # Dùng 2 nhân CPU để load dữ liệu nền

# Kiểm tra xem máy có GPU không để bật chế độ FP16
HAS_GPU = torch.cuda.is_available()
USE_FP16 = True if HAS_GPU else False 
# ==========================================

label_map = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}

def main():
    print(f"🚀 Bắt đầu Fine-tuning trên thiết bị: {'GPU (Nhanh)' if HAS_GPU else 'CPU (Chậm)'}...")
    
    # 1. LOAD DATA
    try:
        train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
        test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
    except FileNotFoundError:
        print("❌ Lỗi: Chưa có file data. Hãy chạy preprocess.py trước!")
        return

    # 2. TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_LEN
        )

    print("⏳ Đang Tokenize dữ liệu...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # 3. MODEL
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    if HAS_GPU: model.to("cuda")

    # 4. TRAINING ARGUMENTS (Đã tối ưu)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        
        # --- Cấu hình tối ưu tốc độ ---
        per_device_train_batch_size=BATCH_SIZE, # Batch nhỏ cho GPU yếu
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION, # Tích lũy để ổn định việc học
        fp16=USE_FP16,                          # [QUAN TRỌNG] Bật chế độ tăng tốc Mixed Precision
        dataloader_num_workers=NUM_WORKERS,     # Load dữ liệu đa luồng
        # -----------------------------
        
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1, # Chỉ lưu 1 model tốt nhất để tiết kiệm ổ cứng
        logging_dir='./logs',
        logging_steps=50,   # In log thường xuyên hơn để theo dõi
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 5. RUN
    print("\n🏋️ Bắt đầu train... (Theo dõi cột 'Loss' giảm dần là tốt)")
    trainer.train()

    print("\n📊 Đánh giá kết quả...")
    eval_result = trainer.evaluate()
    print(f"🎯 Accuracy: {eval_result['eval_accuracy']:.4f}")

    print("💾 Đang lưu model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Đã xong!")

if __name__ == "__main__":
    main()