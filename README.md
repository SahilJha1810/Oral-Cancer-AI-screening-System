# 🩺 Oral Cancer AI Screening System

**An end-to-end clinical decision-support system that screens intraoral photographs for oral squamous cell carcinoma (OSCC) using an EfficientNet ensemble, with Grad-CAM explainability and a full patient-facing web app.**

> ⚠️ \*\*Medical disclaimer:\*\* This system is a research prototype and screening aid. It is \*\*not\*\* a diagnostic device and has not been clinically validated or approved by any regulatory body. It must never replace evaluation by a licensed medical professional.



## 📌 Table of Contents

1. [Problem \& Impact](#-problem--impact)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [AI Model](#-ai-model)
5. [Dataset](#-dataset)
6. [Results](#-results)
7. [Screenshots](#-screenshots)
8. [Tech Stack](#-tech-stack)
9. [Research Paper](#-research-paper)
10. [Installation](#-installation)
11. [License](#-license)



## 🎯 Problem \& Impact

Oral squamous cell carcinoma (OSCC) is the most common type of oral cancer and a major global health concern. Delayed diagnosis remains one of the leading causes of poor survival outcomes, as early-stage oral lesions often resemble benign conditions during routine oral examinations. In many regions, limited access to oral medicine specialists and oral pathologists means that accurate diagnosis frequently depends on the experience of the examining clinician.

This project tackles that gap with an **automated, interpretable screening pipeline** that:

* Classifies intraoral clinical photographs as malignant or non-malignant in seconds.
* Prioritizes **sensitivity** (minimizing missed cancers) over raw accuracy, because in cancer screening a false negative is far more costly than a false positive.
* Shows *why* it made a decision via Grad-CAM heatmaps, instead of acting as an unexplainable black box — a key requirement for clinical trust.
* Wraps the model in a real, deployable product (auth, patient history, reports, care guidance, chat) rather than a notebook-only proof of concept.

The long-term goal is a low-cost, point-of-care screening tool that can flag high-risk lesions for further professional evaluation, particularly in settings with limited access to oral oncology specialists.



## ✨ Key Features

* 📸 **AI-powered scan** — upload/capture an intraoral photo and get an instant Cancer / Non-Cancer classification.
* 🎯 **Risk stratification** — cancer probability is mapped to Low / Medium / High risk bands with a corresponding clinical recommendation.
* 🔥 **Grad-CAM explainability** — every prediction ships with a heatmap overlay showing which regions of the image drove the decision.
* 🧑‍⚕️ **Patient \& scan history** — track scans over time per patient, view trends on a dashboard, and generate shareable reports.
* 💬 **AI health assistant (chat)** — a guarded, context-aware chatbot (via Groq/Llama 3) that can answer questions about a patient's own scan results and general oral-health guidance, with input-sanitization and prompt-injection guardrails.
* 💡 **Daily oral-health tips** — LLM-generated, risk-personalized daily tips.
* 🏥 **Nearby clinic lookup** — surfaces nearby dental/oncology care options for follow-up.
* 🔐 **Secure auth \& storage** — Supabase-backed authentication, row-level data isolation, and signed URLs for scan images.



## 🏗 System Architecture

```
┌──────────────────────┐        HTTPS/REST        ┌───────────────────────────┐
│   React + Vite SPA   │ ────────────────────────▶ │       FastAPI Backend     │
│  (frontend/)         │ ◀──────────────────────── │       (backend/)          │
│  - Dashboard          │                           │  Routers: predict, chat,  │
│  - New Scan           │                           │  history, report, patients,│
│  - History / Reports  │                           │  dashboard, clinics, tips  │
│  - Care Guidance      │                           └─────────────┬─────────────┘
│  - Chat               │                                         │
└──────────────────────┘                                         │
                                                                   ▼
                                        ┌────────────────────────────────────────┐
                                        │        Inference Pipeline (PyTorch)     │
                                        │  Preprocess → EfficientNet-B1 + B2      │
                                        │  ensemble → softmax avg → Grad-CAM      │
                                        └───────────────────┬──────────────────--┘
                                                              │
                     ┌────────────────────────────────────────┼───────────────────┐
                     ▼                                        ▼                   ▼
            ┌─────────────────┐                     ┌──────────────────┐  ┌──────────────┐
            │  Supabase (Auth,  │                     │  Supabase Storage │  │  Groq LLM API │
            │  Postgres DB)     │                     │  (scan images,    │  │  (chat + daily│
            │                   │                     │   Grad-CAM PNGs)  │  │   tips)       │
            └─────────────────┘                     └──────────────────┘  └──────────────┘
```

**Request flow for a scan:**

1. User uploads a photo from the React frontend.
2. FastAPI validates the file, preprocesses it (resize to 224×224, ImageNet normalization).
3. The image is run through both EfficientNet-B1 and EfficientNet-B2; their softmax outputs are averaged for the final prediction.
4. Grad-CAM generates a heatmap overlay from the B1 model's activations.
5. Prediction, confidence, risk level, recommendation, and heatmap are persisted to Supabase and returned to the client.
6. The dashboard, history, and chat features all read from this same prediction record.



## 🧠 AI Model

The production backend serves an **ensemble of EfficientNet-B1 and EfficientNet-B2** (averaged softmax probabilities) for the final Cancer / Non-Cancer decision, backed by the research and experimentation documented in the accompanying paper.

|Component|Details|
|-|-|
|Backbone|EfficientNet (B0 baseline, B1 production, B2 comparison) via `timm`, pretrained on ImageNet|
|Task|Binary classification — malignant (OSCC) vs. non-malignant|
|Input|224×224 RGB, ImageNet-normalized|
|Fine-tuning|All layers fine-tuned at a low learning rate (found to outperform freezing the first 80% of layers)|
|Regularization|Dropout (0.3) before the final dense layer, L2 weight regularization (λ = 1e-5)|
|Explainability|Grad-CAM heatmaps generated from the final convolutional layer|
|Serving|Both B1 and B2 weights loaded at FastAPI startup; ensemble averaging at inference time|

**Why an ensemble, and why B1 over B0/B2?** The accompanying research paper found a *non-linear* relationship between model capacity and diagnostic performance: EfficientNet-B0 (baseline) reached \~85% test accuracy, EfficientNet-B1 jumped to \~99% accuracy with far fewer false negatives, and EfficientNet-B2 showed early signs of overfitting with a slight performance dip. B1 was identified as the capacity "sweet spot," which is why it anchors the production ensemble.



## 📊 Dataset

* **Size:** 2,314 de-identified, multi-source intraoral clinical images.
* **Class balance:** 1,269 malignant (OSCC) / 1,045 non-cancerous.
* **Splits:** Stratified 70% train / 15% validation / 15% test, with strict de-duplication checks to prevent data leakage across splits.
* **Preprocessing:** Resize to 224×224, ImageNet-statistics normalization.
* **Augmentation:** Random resized cropping, horizontal flips, minor rotations (±15°), and color jitter (brightness/contrast/saturation) to simulate real-world clinical photography conditions (inconsistent lighting, angles, devices).

> The dataset itself is not distributed in this repository due to patient data sensitivity. See \[Installation](#-installation) for how to plug in your own trained weights.



## 📈 Results

|Model|Test Accuracy|ROC-AUC|Notes|
|-|-|-|-|
|EfficientNet-B0 (baseline)|\~85%|0.997|Recall (sensitivity) 85.7%, precision 70.6%, F1 77.4|
|**EfficientNet-B1 (production)**|**\~99%**|**>0.99**|Best trade-off; false negatives dropped from 14 (B0) to 4|
|EfficientNet-B2|\~97%|>0.99|Slight rise in false negatives — early sign of overfitting|

* **Confusion matrix (B0 baseline):** most malignant cases correctly identified, with the small number of misses concentrated in early-stage or visually subtle lesions — cases that are also difficult for experienced clinicians to catch on sight.
* **Grad-CAM validation:** heatmaps consistently localize on the lesion itself for malignant cases, with no spurious high-activation regions on benign images — supporting that the model is learning pathology, not background artifacts.
* **Design takeaway:** the study found that *moderate* scaling (B0 → B1) captured meaningful additional signal, while further scaling (B1 → B2) began to overfit on a relatively small, specialized medical dataset — reinforcing that architecture selection should be evidence-driven, not "bigger is always better."

Full methodology, comparative literature review, and error analysis are in the [research paper](#-research-paper).



## 🖼 Screenshots


|Screen|Description|
|-|-|
|`docs/screenshots/Dashboard.jpg`|Home dashboard — scan trends, last scan summary, daily tip|
|`docs/screenshots/Profile-Dashboard.jpg`|Profile Dashboard|
|`docs/screenshots/Result.jpg`|Prediction result with risk level + Grad-CAM heatmap|
|`docs/screenshots/Scan-History.jpg`|Scan history / patient timeline|
|`docs/screenshots/Oral AI Assistant.jpg`|AI health assistant chat|
|`docs/screenshots/Roc Curve.png`|ROC Curve|
|`docs/screenshots/Confusion Matrix.jpg`|Model prediction accuracy breakdown|
|`docs/screenshots/Grad Cam Visualization.jpg`|Lesion-focused model explainability heatmaps|



## 🛠 Tech Stack

**Frontend**

* React 18 + Vite
* React Router
* Tailwind CSS
* Axios
* Supabase JS client
* lucide-react (icons)

**Backend**

* FastAPI + Uvicorn
* PyTorch 2.7 + `timm` (EfficientNet-B1/B2)
* OpenCV, Pillow, NumPy (preprocessing \& Grad-CAM)
* Supabase (Postgres, Auth, Storage)
* Groq API (Llama 3) for chat + daily tips
* Pytest for backend tests



## 📄 Research Paper

This project is grounded in the accompanying paper:

**"Automated Oral Cancer Screening from Clinical Photographs Using EfficientNet-Based Transfer Learning"**
*Sahil Jha, Harsh Tomar — Department of Computer Science and Engineering, Lovely Professional University, Punjab, India*

The paper covers the full methodology: dataset curation, EfficientNet-B0/B1/B2 comparative scaling analysis, Grad-CAM-based interpretability, ROC/confusion-matrix evaluation, and a literature review situating this work against prior OSCC detection studies.


```markdown
[Read the full paper (PDF)](docs/paper/oral-cancer-efficientnet.pdf)
```

If you use this work academically, please cite it — see [Citation](#-citation) below once you have finalized publication details (conference/journal, year, DOI).



## ⚙️ Installation

### Prerequisites

* Python 3.10+
* Node.js 18+
* A [Supabase](https://supabase.com/) project (Auth + Postgres + Storage)
* A [Groq](https://groq.com/) API key (for chat \& daily tips)
* Trained model weights: `b1.pth` and `b2.pth` (not included in this repo — see note below)

### 1\. Clone the repo

```bash
git clone https://github.com/<your-username>/oral-cancer-ai-screening-system.git
cd oral-cancer-ai-screening-system
```

### 2\. Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

cp .env.example .env
# Fill in SUPABASE\_URL, SUPABASE\_ANON\_KEY, SUPABASE\_SERVICE\_ROLE\_KEY,
# CORS\_ORIGINS, GROQ\_API\_KEY, GROQ\_MODEL

# Place trained weights here:
mkdir -p models
# models/b1.pth
# models/b2.pth

uvicorn app.main:app --reload --port 8000
```

Set up the database schema and storage buckets using `backend/schema.sql` and `SUPABASE\_SETUP.md`.

### 3\. Frontend setup

```bash
cd frontend
npm install

cp .env.example .env
# Fill in VITE\_SUPABASE\_URL, VITE\_SUPABASE\_ANON\_KEY, VITE\_API\_BASE\_URL

npm run dev
```

The app will be available at `http://localhost:5173`, talking to the API at `http://127.0.0.1:8000`.

### 4\. Run backend tests

```bash
cd backend
pytest
```

> \*\*Note on model weights:\*\* `backend/models/\*.pth` is gitignored (see `.gitignore`) since trained weights on clinical data should not be committed to a public repo, especially without a data-use agreement. Either train your own using the methodology in the \[paper](#-research-paper), or host weights externally (e.g. Hugging Face Hub, Git LFS, or a release asset) and document the download step here.



## 📜 License

This project is licensed under the [MIT License](LICENSE).



## 🙌 Acknowledgements

Built by **Sahil Jha**, and **Harsh Tomar** — Department of Computer Science and Engineering, Lovely Professional University, Punjab, India.

