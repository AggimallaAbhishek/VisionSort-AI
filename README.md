# 🚀 VisionSort AI  
### Intelligent Photo Quality Assessment & Automatic Image Curation System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Supabase](https://img.shields.io/badge/Supabase-Database%20%26%20Storage-3ECF8E)
![Vercel](https://img.shields.io/badge/Vercel-Frontend-black)

VisionSort AI is a full-stack web application that automatically analyzes and sorts uploaded photos based on image quality metrics such as blur detection, brightness evaluation, duplicate removal, and optional AI-based classification.

The system combines traditional Computer Vision techniques with Deep Learning to intelligently filter out low-quality images and return only the best photos.

---

## ✨ Features

- 📤 Multi-image upload support
- 🔍 Blur detection using Variance of Laplacian
- 🌗 Brightness & exposure analysis
- ♻ Duplicate image detection (perceptual hashing)
- 🤖 Optional CNN-based image quality classification
- ☁ Cloud storage with Supabase
- ⚡ Full-stack deployment (Frontend + Backend separated)

---

## 🏗 System Architecture

```
Frontend (Vercel)
        ↓
FastAPI Backend (Render/Railway)
        ↓
Image Processing (OpenCV + PyTorch)
        ↓
Supabase Storage + PostgreSQL
        ↓
Sorted Results Returned to User
```

---

## 🛠 Tech Stack

### Backend
- Python
- FastAPI
- OpenCV
- PyTorch
- Pillow
- NumPy

### Frontend
- HTML/CSS / React / Next.js

### Database & Storage
- Supabase (PostgreSQL + Storage Buckets)

### Deployment
- Vercel (Frontend)
- Render or Railway (Backend)

---

## 📂 Project Structure

```
vision-sort-ai/
│
├── backend/
│   ├── main.py
│   ├── utils/
│   │   ├── blur_detection.py
│   │   ├── brightness_check.py
│   │   ├── duplicate_check.py
│   │   └── model_predict.py
│   ├── model/
│   │   └── photo_model.pth
│   └── requirements.txt
│
├── frontend/
│   ├── index.html / React files
│   └── styles.css
│
└── README.md
```

---

## 🔧 Installation (Backend Setup)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/vision-sort-ai.git
cd vision-sort-ai/backend
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install fastapi uvicorn opencv-python pillow numpy torch supabase imagehash
```

---

## ▶️ Run Backend Server

```bash
uvicorn main:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

---

## 🌐 Frontend Setup

If using React / Next.js:

```bash
npm install
npm run dev
```

If using static HTML:

Simply open `index.html` or deploy to Vercel.

---

## 🧠 How Blur Detection Works

Blur detection is implemented using the Variance of Laplacian method:

```python
variance = cv2.Laplacian(gray_image, cv2.CV_64F).var()
```

- Low variance → Blurry image  
- High variance → Sharp image  

---

## 🤖 AI Model (Optional)

A pretrained CNN (e.g., ResNet / MobileNet) can be fine-tuned for image quality classification.

Steps:
1. Prepare labeled dataset
2. Train model
3. Save `.pth` file
4. Load model during inference
5. Predict quality class

---

## 📊 Use Cases

- Photography workflow automation
- Bulk image cleaning
- Dataset preprocessing for ML models
- Event photo selection
- Research in image quality assessment

---

## 🚀 Deployment

### Backend
- Deploy using Render / Railway
- Add environment variables:
  - SUPABASE_URL
  - SUPABASE_KEY

### Frontend
- Deploy on Vercel
- Set backend API URL

---

## 🔮 Future Improvements

- Aesthetic score prediction
- Face-aware ranking system
- Similar image clustering
- GPU acceleration
- SaaS version with authentication

---

## 📄 License

MIT License

---

## 👨‍💻 Author

Aggimalla Abhishek  
DSAI | Computer Vision | AI Systems
