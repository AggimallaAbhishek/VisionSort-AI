# VisionSort AI

VisionSort AI is a full-stack image quality pipeline that accepts batch uploads and automatically categorizes photos into:

- `good`
- `blurry`
- `dark`
- `overexposed`
- `duplicates`

Architecture:

Frontend (Vercel/static)
-> FastAPI Backend (Render/Railway/EC2)
-> Image Processing (OpenCV + optional PyTorch)
-> AWS S3 + AWS RDS PostgreSQL
-> JSON response to frontend

## Project Structure

```text
vision-sort-ai/
├── render.yaml
├── backend/
│   ├── main.py
│   ├── aws_client.py
│   ├── requirements.txt
│   ├── runtime.txt
│   ├── .env.example
│   ├── utils/
│   │   ├── blur_detection.py
│   │   ├── brightness_check.py
│   │   ├── duplicate_check.py
│   │   └── model_predict.py
│   └── model/
│       └── photo_model.pth
├── frontend/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── vercel.json
└── README.md
```

## Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Run locally:

```bash
uvicorn main:app --host 0.0.0.0 --port 10000 --reload
```

Production start command:

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

## Required AWS Resources

1. S3 buckets
- `S3_UPLOADS_BUCKET` for original uploads
- `S3_PROCESSED_BUCKET` for resized processed images

2. RDS PostgreSQL table `images`

```sql
create extension if not exists pgcrypto;

create table if not exists images (
  id uuid primary key default gen_random_uuid(),
  user_id text not null default 'anonymous',
  file_name text not null,
  blur_score double precision,
  brightness_level text,
  ai_label text,
  final_status text check (final_status in ('good','blurry','dark','overexposed','duplicates')),
  created_at timestamptz not null default now()
);
```

## API

### `GET /`
Returns backend and service readiness status.

### `POST /upload`
Multipart form-data with repeated field name `files`.

Example:

```bash
curl -X POST "http://localhost:10000/upload" \
  -F "files=@/absolute/path/photo1.jpg" \
  -F "files=@/absolute/path/photo2.jpg"
```

Response shape:

```json
{
  "good": [],
  "blurry": [],
  "dark": [],
  "overexposed": [],
  "duplicates": []
}
```

Each item includes:

- `file_name`
- `blur_score`
- `brightness_level`
- `ai_label`
- `final_status`
- `preview_data_url`
- `storage_path` (original object)
- `processed_storage_path` (resized object)

## Frontend Setup

Serve locally:

```bash
cd frontend
python3 -m http.server 5500
```

The frontend reads backend URL in this order:

1. `window.VISIONSORT_API_BASE_URL`
2. `<meta name="visionsort-api-base-url" ...>` in `frontend/index.html`
3. fallback `http://localhost:10000`

## Train MobileNetV2 (Optional AI Label)

`model_predict.py` is located at:

- `backend/utils/model_predict.py`

To train and generate `backend/model/photo_model.pth`, use:

```bash
cd backend
source .venv/bin/activate
python train_mobilenetv2.py \
  --data-root ../training_data \
  --epochs 24 \
  --batch-size 16 \
  --freeze-features \
  --unfreeze-epoch 5 \
  --patience 6
```

This training script now includes:

- stronger augmentation
- class-balanced loss weights
- staged fine-tuning (head first, then backbone)
- early stopping
- per-class validation metrics

Expected dataset structure:

```text
training_data/
├── train/
│   ├── good/
│   ├── blurry/
│   ├── dark/
│   ├── overexposed/
│   └── duplicates/
└── val/
    ├── good/
    ├── blurry/
    ├── dark/
    ├── overexposed/
    └── duplicates/
```

After training:

1. Ensure `backend/model/photo_model.pth` is non-empty.
2. Set `ENABLE_AI_LABEL=true` in backend env.
3. Restart backend and test upload.

Optional AI + rule fusion knobs:

- `AI_ASSISTED_STATUS=true` enable AI-assisted final status calibration
- `AI_MIN_CONFIDENCE=0.70` minimum confidence for AI to affect status
- `AI_PROMOTE_GOOD_CONFIDENCE=0.90` required confidence to promote a rule issue to `good`
- `AI_BORDERLINE_FACTOR=0.90` how close to blur threshold a sample must be for AI promotion
- `DARK_PROMOTE_MIN_BRIGHTNESS=45` dark-to-good promotion lower bound
- `OVEREXPOSED_PROMOTE_MAX_BRIGHTNESS=210` overexposed-to-good promotion upper bound

## Deploy

### Render backend

1. Create a new Blueprint on Render from this repo.
2. Render reads `render.yaml` and creates `visionsort-ai-backend`.
3. Set secret env vars in Render dashboard:
- `AWS_REGION`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_UPLOADS_BUCKET`
- `S3_PROCESSED_BUCKET`
- `DATABASE_URL`
- `ALLOWED_ORIGINS` (include your Vercel domain)
- `ALLOWED_ORIGIN_REGEX` (recommended: `^https://.*\\.vercel\\.app$`)
- `PERSIST_TASK_TIMEOUT_SECONDS` (recommended: `30`)
- `PERSIST_OVERALL_TIMEOUT_SECONDS` (recommended: `90`)
- `S3_CONNECT_TIMEOUT_SECONDS` (recommended: `3`)
- `S3_READ_TIMEOUT_SECONDS` (recommended: `12`)
- `S3_MAX_ATTEMPTS` (recommended: `2`)
- `DB_CONNECT_TIMEOUT_SECONDS` (recommended: `8`)
- `DB_STATEMENT_TIMEOUT_MS` (recommended: `15000`)

### Vercel frontend

1. Import the same repo in Vercel.
2. Set **Root Directory** to `frontend`.
3. Deploy.
4. Update `frontend/index.html` meta tag `visionsort-api-base-url` to your Render backend URL.

## Deployment Notes

- Ensure `ALLOWED_ORIGINS` includes local and deployed frontend domains.
- Backend validates max file count, file size, and allowed image types.
- Images are resized using `MAX_IMAGE_WIDTH` before analysis.
- Response previews are compressed using `PREVIEW_MAX_WIDTH` and `PREVIEW_JPEG_QUALITY` to keep large batches stable.
- If model weights are missing/untrained, `ai_label` is `model_unavailable`.
- For large uploads, prefer async endpoint `POST /upload/async` and poll `GET /jobs/{job_id}`.
- Use `GET /` to confirm timeout config values after Render deploy.
- Frontend batch controls (in `frontend/index.html` meta tags):
  - `visionsort-max-request-size-mb` (default `35`)
  - `visionsort-max-batch-files` (default `10`)
