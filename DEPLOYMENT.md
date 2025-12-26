# ML Service Deployment Guide

## Quick Deploy to Railway

1. **Install Railway CLI**

   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway**

   ```bash
   railway login
   ```

3. **Deploy from ml_service directory**

   ```bash
   cd ml_service
   railway init
   railway up
   ```

4. **Set Environment Variables** (in Railway dashboard)
   - `PORT=5000` (automatically set)
   - Add any custom env vars if needed

Railway will automatically:

- Detect Python project
- Install Tesseract OCR via nixpacks.toml
- Install Python dependencies
- Deploy your service

## Docker Deployment

**Build:**

```bash
docker build -t ml-service .
```

**Run:**

```bash
docker run -p 5000:5000 ml-service
```

**Deploy to Docker Hub:**

```bash
docker tag ml-service yourusername/ml-service
docker push yourusername/ml-service
```

## Render Deployment

1. Go to [render.com](https://render.com) and create Web Service from GitHub
2. Settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python -m app.main`
   - Render automatically sets `PORT` environment variable
3. Render auto-deploys on push

## Production Checklist

- [ ] Update CORS origins in main.py for production domain
- [ ] Set proper environment variables
- [ ] Configure logging level
- [ ] Set up monitoring (Railway/Render have built-in)
- [ ] Test endpoints after deployment
- [ ] Update backend to use production ML service URL

## Health Check

After deployment, test:

```bash
curl https://your-ml-service.railway.app/
```

Should return:

```json
{ "message": "Certificate Verification ML Service", "status": "online" }
```
