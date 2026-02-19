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
├── backend/
│   ├── main.py
│   ├── aws_client.py
│   ├── requirements.txt
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
│   ├── style.css
│   └── script.js
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

Update `.env` values for your AWS account and database.

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
- `S3_PROCESSED_BUCKET` for resized processed images (optional but recommended)

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
- `storage_path` (original object, if upload enabled)
- `processed_storage_path` (resized object, if processed bucket configured)

## Frontend Setup

Deploy `frontend/` as static files on Vercel (or any static host).

To use a deployed backend URL, set this before loading `script.js` in `index.html`:

```html
<script>
  window.VISIONSORT_API_BASE_URL = "https://your-backend-domain";
</script>
```

## Deployment Notes

- Ensure `ALLOWED_ORIGINS` includes your frontend domain(s).
- Backend validates file count, file size, supported image types, and invalid images.
- Images are resized to `MAX_IMAGE_WIDTH` before analysis.
- If the model file is missing/untrained, `ai_label` returns `model_unavailable`.
