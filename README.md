VisionSort AI – Intelligent Photo Quality Assessment & Curation System

VisionSort AI is a full-stack web application that automatically analyzes and sorts uploaded photos based on image quality metrics such as blur detection, brightness evaluation, duplicate removal, and optional AI-based classification.

The system uses traditional computer vision techniques combined with deep learning to filter out blurry, dark, overexposed, or duplicate images and returns only the best-quality photos.

This project demonstrates the integration of:

Computer Vision (OpenCV)

Deep Learning (PyTorch)

Backend APIs (FastAPI)

Cloud Storage & Database (Supabase)

Frontend Deployment (Vercel)

🧠 Key Features

📤 Multi-image upload support

🔍 Blur detection using Variance of Laplacian

🌗 Brightness and exposure analysis

♻ Duplicate image detection using perceptual hashing

🤖 Optional CNN-based quality classification

☁ Cloud storage integration with Supabase

⚡ Deployed full-stack architecture

🏗 System Architecture
Frontend (Vercel)
        ↓
FastAPI Backend (Render/Railway)
        ↓
Image Processing (OpenCV + PyTorch)
        ↓
Supabase Storage + PostgreSQL
        ↓
Sorted Results Returned to User

🛠 Tech Stack

Backend

Python

FastAPI

OpenCV

PyTorch

Frontend

HTML/CSS / React / Next.js

Database & Storage

Supabase (PostgreSQL + Storage)

Deployment

Vercel (Frontend)

Render/Railway (Backend)

🎯 Use Cases

Photography workflow automation

Bulk photo cleanup

Dataset preprocessing for ML

Event photography filtering

AI-based image quality research

📊 Future Improvements

Aesthetic scoring model

Face-aware ranking

Similar image clustering

GPU optimization

SaaS version with authentication
