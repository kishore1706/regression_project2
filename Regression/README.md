# Refinance Volume Prediction App

A Streamlit web app that predicts refinance volume based on 11 economic indicators using a trained Random Forest model.

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run frontend.py
```

Navigate to http://localhost:8501 in your browser.

## Deployment Options

### Option 1: Streamlit Community Cloud (Easiest)

1. Create a GitHub repo with this code:
   ```bash
   git init
   git add .
   git commit -m "initial commit"
   git push origin main
   ```

2. Go to https://share.streamlit.io

3. Click "New app", select your repo, branch (`main`), and file (`frontend.py`)

4. Deploy! Your app is now public at `https://share.streamlit.io/<username>/<repo-name>`

**Note:** If model files are large, upload them to cloud storage (S3/GCS) and load them via URL.

---

### Option 2: Render (Container-based, Free Tier Available)

1. Push code to GitHub (include `Dockerfile` and `requirements.txt`)

2. Go to https://render.com, sign up, and create new **Web Service**

3. Connect your GitHub repo

4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run frontend.py --server.port 8501 --server.headless true`
   - **Instance Type:** Free (suitable for light traffic)

5. Deploy! Your app runs at `https://<service-name>.onrender.com`

---

### Option 3: Docker + Google Cloud Run (Scalable)

1. Build image:
   ```bash
   docker build -t gcr.io/PROJECT_ID/streamlit-app:latest .
   docker push gcr.io/PROJECT_ID/streamlit-app:latest
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy streamlit-app \
     --image gcr.io/PROJECT_ID/streamlit-app:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 512Mi \
     --cpu 1
   ```

3. App accessible at the provided Cloud Run URL

---

### Option 4: Heroku (Deprecated but still works with buildpacks)

1. Install Heroku CLI and login:
   ```bash
   heroku login
   ```

2. Create app:
   ```bash
   heroku create <app-name>
   ```

3. Add buildpack:
   ```bash
   heroku buildpacks:add heroku/python
   ```

4. Push to Heroku:
   ```bash
   git push heroku main
   ```

5. View logs:
   ```bash
   heroku logs --tail
   ```

---

## Model & Scaler Storage

**For production:**

- **Small files** (<50MB): Commit to repo (already done)
- **Large files**: Upload to cloud storage:
  - AWS S3
  - Google Cloud Storage
  - Azure Blob Storage

Example (load from S3):
```python
import boto3
import joblib

s3 = boto3.client('s3')
s3.download_file('my-bucket', 'fmodel.sav', 'fmodel.sav')
model = joblib.load('fmodel.sav')
```

---

## Environment Variables (Optional)

Add secrets in deployment platform:

```toml
# .streamlit/secrets.toml (local only, never commit)
[aws]
access_key = "your_key"
secret_key = "your_secret"
bucket = "my-bucket"
```

Access in app:
```python
import streamlit as st
secret = st.secrets["aws"]["access_key"]
```

---

## Monitoring & Logs

- **Streamlit Cloud:** Built-in logs in dashboard
- **Render:** Logs tab in service details
- **Cloud Run:** View in Cloud Logging console
- **Heroku:** `heroku logs --tail`

---

## Input Features (11 Economic Indicators)

1. Month No
2. Mortgage Rate
3. Inflation
4. Housing Price Index
5. Treasury Yield
6. Unemployment Rate
7. GDP
8. Business Confidence Index
9. Consumer Confidence Index
10. Initial Unemployment Claim
11. Disposable Income

---

## Testing

Test your deployed app with example input values from `final_predictions.xlsx`.

---

## Support

For issues:
- Streamlit docs: https://docs.streamlit.io
- Render docs: https://render.com/docs
- Google Cloud Run: https://cloud.google.com/run/docs
