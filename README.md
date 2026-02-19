# VisionSort AI

VisionSort AI is a full-stack image quality pipeline that accepts batch uploads and automatically categorizes photos into:

- `good`
- `blurry`
- `dark`
- `overexposed`
- `duplicates`

It uses FastAPI + OpenCV + optional PyTorch inference, stores assets/metadata in Supabase, and returns structured JSON for a static frontend.

## Project Structure

```text
vision-sort-ai/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── supabase_client.py
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
```

Create `.env` in `backend/`:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_or_anon_key
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
DEFAULT_USER_ID=anonymous
BLUR_THRESHOLD=100
DUPLICATE_HASH_DISTANCE=5
MAX_IMAGE_WIDTH=1024
MAX_FILE_SIZE_MB=10
ENABLE_AI_LABEL=true
```

Run backend locally:

```bash
uvicorn main:app --host 0.0.0.0 --port 10000 --reload
```

Production start command:

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

## Frontend Setup

You can deploy `frontend/` as static files (Vercel/Netlify/etc.).

To use a non-local backend, set before loading `script.js`:

```html
<script>
  window.VISIONSORT_API_BASE_URL = "https://your-backend-domain";
</script>
```

Then open `frontend/index.html` or deploy the folder.

## API Contract

### `GET /`
Health check.

### `POST /upload`
Multipart form-data with repeated field name `files`.

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
- `storage_path`

## Supabase Requirements

1. Create table `images` with columns:
   - `id` (uuid)
   - `user_id` (text/uuid)
   - `file_name` (text)
   - `blur_score` (float)
   - `brightness_level` (text)
   - `ai_label` (text)
   - `final_status` (text)
   - `created_at` (timestamp)
2. Create storage buckets:
   - `uploads`
   - `processed`

## Notes

- Duplicate detection uses perceptual hashing (`imagehash.phash`) with configurable Hamming threshold.
- AI model loading is optional. If `photo_model.pth` is missing or untrained, prediction returns `model_unavailable`.
- Invalid types, empty files, and oversized images are skipped safely.
