import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup, logging
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

def main():
    logging.set_verbosity_error()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용하는 장치 : ", device)

    path = "mlb_labeled_data.csv"
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except FileNotFoundError:
        print(f"오류: {path} 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    text = list(df["text"].values)
    labels = df["label"].values

    print("\n=== MLB 데이터 확인 ===")
    print(" 문장 : ", text[:5])
    print(" 라벨 : ", labels[:5])

    tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
    inputs = tokenizer(text, truncation=True, max_length=512, add_special_tokens=True, padding="max_length")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    num_to_print = 3
    print("\n=== 토큰화 샘플 ===")
    for j in range(num_to_print):
        print(f"\n{j+1}번째 데이터")
        print("토큰: ", input_ids[j])
        print("어텐션 마스크: ", attention_mask[j])

    tx, vx, ty, vy = train_test_split(input_ids, labels, test_size=0.2, random_state=2026)
    tm, vm, _, _ = train_test_split(attention_mask, labels, test_size=0.2, random_state=2026)

    batch_size = 8

    train_inputs = torch.tensor(tx)
    train_labels = torch.tensor(ty)
    train_masks = torch.tensor(tm)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_inputs = torch.tensor(vx)
    valid_labels = torch.tensor(vy)
    valid_masks = torch.tensor(vm)
    valid_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epoch = 4
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * epoch)

    epoch_results = []

    for e in range(epoch):
        model.train()
        total_train_loss = 0.0

        process_bar = tqdm(train_dataloader, desc=f"Training epoch {e+1}", leave=False)

        for batch in process_bar:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_masks, batch_labels = batch
            model.zero_grad()

            outputs = model(batch_ids, attention_mask=batch_masks, labels=batch_labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            process_bar.set_postfix({'loss': loss.item()})

        model.eval()
        train_preds, train_true = [], []

        progress_bar_t = tqdm(train_dataloader, desc=f"Evaluation Train Epoch {e+1}", leave=False)
        for batch in progress_bar_t:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_masks, batch_labels = batch
            with torch.no_grad():
                outputs = model(batch_ids, attention_mask=batch_masks)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(batch_labels.cpu().numpy())
        train_acc = np.sum(np.array(train_preds) == np.array(train_true)) / len(train_preds)

        valid_preds, valid_true = [], []

        progress_bar_v = tqdm(valid_dataloader, desc=f"Evaluation Epoch {e+1}", leave=False)
        for batch in progress_bar_v:
            batch = tuple(t.to(device) for t in batch)
            batch_ids, batch_masks, batch_labels = batch
            with torch.no_grad():
                outputs = model(batch_ids, attention_mask=batch_masks)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            valid_preds.extend(preds.cpu().numpy())
            valid_true.extend(batch_labels.cpu().numpy())
        valid_acc = np.sum(np.array(valid_preds) == np.array(valid_true)) / len(valid_preds)

        epoch_results.append([avg_train_loss, train_acc, valid_acc])

    print("\n=== 학습 및 검증 결과 ===")
    for idx, (loss, tacc, vacc) in enumerate(epoch_results, start=1):
        print(f"Epoch {idx}: 학습오차 - {loss:.4f}, 학습정확도 - {tacc:.4f}, 검증정확도 - {vacc:.4f}")

    print("\n=== 모델 저장 ===")
    model_save_path = "mobilebert_mlb_sentiment.pt"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"모델 저장 완료: {model_save_path}")

if __name__ == "__main__":
    main()