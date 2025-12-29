from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import prediction, metrics
import os

app = FastAPI(
    title="NeuroScan AI - Brain Tumor Detection API",
    description="AI-powered brain tumor detection with explainable AI",
    version="1.0.0",
)

# CORS middleware - Allow Vercel frontend and local development
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",  # Vite default port
    "https://*.vercel.app",  # Your Vercel deployments
]

# Add production frontend URL from environment variable
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex="https://.*\.vercel\.app",  # Allow all Vercel preview deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router)
app.include_router(metrics.router, prefix="/api", tags=["metrics"])


@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "NeuroScan AI - Brain Tumor Detection API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "predict": "/api/predict",
            "gradcam": "/api/gradcam"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}
