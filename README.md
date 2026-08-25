# DSC2026# LegalIR (Task 1) — UIT Data Science Challenge 2026

Truy vấn thông tin pháp luật tiếng Việt: cho một câu hỏi, trả về tối đa 5
`document_id` của các văn bản pháp luật liên quan.

## Dữ liệu

```
data/raw/
  train.json         7,000 câu hỏi có nhãn: {"qid": {"question": ..., "answer": [doc_id, ...]}}
  public_test.json   1,000 câu hỏi cần dự đoán (answer: null)
  corpus.jsonl        8,532 văn bản pháp luật, 1 dòng JSON/văn bản:
                       {"id": ..., "name": ..., "link": ..., "passage": ...}
```

`corpus.jsonl` được gộp lại từ `selected-contexts.zip` (8,532 file
`context_<id>.json`) — dùng JSONL thay vì 1 JSON khổng lồ (488MB) để load
tiết kiệm RAM hơn trên môi trường giới hạn (~4GB RAM, 1 CPU).

## Kiến trúc

```
question
  → tokenize (Vietnamese, accent-stripped, stopword-filtered)
  → BM25 (rank_bm25) trên toàn bộ 8,532 văn bản
     - title (name) được nhân bản x4 lần để boost trọng số so khớp tiêu đề
     - mỗi văn bản bị cắt tối đa 6,000 token khi index (một số văn bản dài
       tới hàng triệu ký tự — cả bộ luật) để giữ RAM/thời gian index trong
       giới hạn phần cứng
  → top-5 doc_id theo BM25 score
```

Đây là retrieval **document-level** (không phải article-level như pipeline
ALQAC) — mỗi record trong corpus là một văn bản pháp luật hoàn chỉnh, không
chia điều/khoản.

## Kết quả (Recall@5 / Precision@5 trên train.json làm dev set)

Chạy `python scripts/evaluate.py --top-k 5` để tái tạo. Kết quả trên toàn
bộ 7,000 câu train.json (dùng làm dev set, xem
`outputs/full_train_eval.json`):

| Metric | Giá trị |
|---|---|
| Recall@5 (chính) | **0.4591** |
| Precision@5 (phụ) | 0.0971 |
| Coverage (≥1 hit) | 0.4739 |

BM25 thuần trên toàn văn bản, chưa dense/hybrid rerank — đây là baseline
mạnh, dễ tái tạo, chạy nhanh trên CPU thường. Xem mục "Hướng cải tiến"
bên dưới để tăng recall.

## Yêu cầu môi trường

```bash
pip install -r requirements.txt
```

Xem `requirements.txt` để biết danh sách và phiên bản thư viện. Không cần
GPU — toàn bộ pipeline chạy trên CPU (đã kiểm chứng trên môi trường 1 CPU,
~4GB RAM).

## Sử dụng

### 0. Chuẩn bị corpus (chỉ cần 1 lần)

`data/raw/corpus.jsonl` (~466MB) không đi kèm gói code vì quá nặng. Build
lại từ `selected-contexts.zip` gốc của đề bài:

```bash
python scripts/build_corpus.py --zip /path/to/selected-contexts.zip
```

### 1. Build / cache BM25 index (tự động lần chạy đầu, ~65s, ~1.9GB RAM)

```bash
python scripts/evaluate.py --top-k 5 --limit 300   # build index + eval nhanh
```

Index được cache tại `outputs/bm25_index.pkl` (~vài trăm MB) — các lần
chạy sau load lại, không phải build lại.

### 2. Evaluate trên train.json (dev set)

```bash
python scripts/evaluate.py --top-k 5                 # toàn bộ 7,000 câu
python scripts/evaluate.py --top-k 5 --limit 500      # nhanh hơn, lấy mẫu
python scripts/evaluate.py --top-k 5 --verbose        # in từng câu
```

### 3. Sinh submission cho public test

```bash
python scripts/predict.py --top-k 5
```

Ghi ra:
- `outputs/submissions/submission.json`
- `outputs/submissions/submission.zip` (nộp file này theo đúng yêu cầu đề bài)

## Ràng buộc đề bài (đã tuân thủ trong code)

- Tối đa 5 `document_id`/câu hỏi (`predict.py` chặn `--top-k > 5`).
- `submission.zip` chỉ chứa `submission.json` ở root, đúng định dạng.

## Hướng cải tiến tiếp theo (chưa làm, do giới hạn phần cứng 1 CPU / 4GB RAM)

- Dense/hybrid reranking (BGE-M3 hoặc Vietnamese embedding) trên top-N BM25
  candidates — cần GPU hoặc CPU mạnh hơn để encode 8,532 văn bản dài.
- Chunk-level indexing cho các văn bản rất dài (whole bộ luật) rồi max-pool
  điểm chunk → văn bản, để không bị cắt mất phần nội dung liên quan nằm
  sau token thứ 6,000.
- Query expansion theo thuật ngữ pháp lý (tương tự
  `src/retrieval/query_expansion.py` trong pipeline ALQAC).