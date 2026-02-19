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

### Vercel frontend

1. Import the same repo in Vercel.
2. Set **Root Directory** to `frontend`.
3. Deploy.
4. Update `frontend/index.html` meta tag `visionsort-api-base-url` to your Render backend URL.

## Deployment Notes

- Ensure `ALLOWED_ORIGINS` includes local and deployed frontend domains.
- Backend validates max file count, file size, and allowed image types.
- Images are resized using `MAX_IMAGE_WIDTH` before analysis.
- If model weights are missing/untrained, `ai_label` is `model_unavailable`.
