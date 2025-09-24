# 🏋️‍♂️ Streamlit — Video Pose Estimation + Squat Counter

A Streamlit app that:
- Uploads a video (`.mp4/.mov/.avi/.mkv`)
- Runs **Ultralytics YOLO Pose** (yolo11 / yolov8)
- Computes **knee angle** (hip–knee–ankle), checks **depth**, and **counts valid squats**
- Writes the **annotated video** with counters and overlays for download

> Works on CPU by default; will use GPU automatically if available and `torch` is CUDA-enabled.

---

## 📁 Project structure

```
.
├── app.py                 # your main Streamlit app (paste the code you have)
├── requirements.txt       # Python deps for Streamlit Cloud / local
└── README.md
```

---

## ⚙️ Local setup (Windows/Mac/Linux)

1) Create & activate a virtual env (Conda or venv). Example with Conda:
```bash
conda create -n pose_app python=3.10 -y
conda activate pose_app
```

2) Install deps:
```bash
pip install -r requirements.txt
```

> If you have an NVIDIA GPU and want CUDA acceleration, install a CUDA-enabled PyTorch that matches your drivers (see PyTorch “Get Started” guide). Otherwise the app runs on CPU just fine.

3) Run:
```bash
streamlit run app.py
```
Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## ☁️ Deploy on Streamlit Cloud

1) Push this folder to a **public GitHub repo** (include `app.py`, `requirements.txt`, and `README.md`).  
2) Go to **share.streamlit.io**, click **“New app”**, pick your repo/branch, and **file path = `app.py`**.  
3) Click **Deploy**. The app will build and then open at a public URL.

**Notes for Streamlit Cloud**
- Builds run on CPU; the app will auto-fallback to `device="cpu"`.
- Video writing uses OpenCV (`mp4v`). If MP4 writing ever fails in your environment, try:
  - Use `.avi` container and `MJPG` fourcc, **or**
  - Add `imageio-ffmpeg` and `av` to requirements and switch to an `imageio`/`PyAV` writer.

---

## 🎛️ App usage

1) Open the app → **Upload a video**  
2) Choose a **model** (e.g., `yolo11n-pose.pt` fastest, `yolo11s-pose.pt` better accuracy).  
3) (Optional) Adjust thresholds:
   - **Stand-up threshold** (default ~165°)
   - **Bottom depth threshold** (default ~100°)
   - **Require hip lower than knee** (on/off)
4) Click **Run Pose + Counter**  
5) Watch the processed video and **download** it.

**Rep logic**
- A **rep** is counted on a full **down → up** cycle **only if** bottom depth is valid:
  - Knee angle ≤ bottom threshold, and
  - (Optional) hip y-position is lower than knee (stricter depth)
- If you come up without valid depth, the rep is **skipped** (shown as “skipped” in HUD).

---

## 🧩 Troubleshooting

- **Pylance / VS Code** can’t find `ultralytics`  
  Make sure the interpreter in VS Code matches the environment where you installed packages:  
  Command Palette → “**Python: Select Interpreter**”.

- **Invalid CUDA 'device=0' requested**  
  Your PyTorch build doesn’t see a GPU. The app already uses:
  ```python
  import torch
  device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
  ```
  If no GPU is detected, it will run on CPU automatically.

- **OpenCV MP4 writing fails**  
  Some environments lack H.264/MP4 encoders. Try:
  1) Switch to `avi`/`MJPG` fourcc in your writer, or  
  2) Add `imageio-ffmpeg` + `av` to `requirements.txt` and use a PyAV/ImageIO writer.

- **Slow inference**  
  - Use a smaller model (`yolo11n-pose.pt` / `yolov8n-pose.pt`)  
  - Increase `frame stride` (e.g., 2 or 3)  
  - Trim your input video

---

## ✅ Requirements

See `requirements.txt`. Minimal core deps:
- `streamlit` — web UI
- `ultralytics` — YOLO Pose models (supports `yolo11*` and `yolov8*`)
- `opencv-python-headless` — video I/O and drawing
- `numpy` — math helpers
- `torch` — deep learning backend (CPU-only by default on Streamlit Cloud)

> Models like `yolo11n-pose.pt` are auto-downloaded by Ultralytics on first run.

---

## 📦 Optional extras

If you want alternate video writers or to debug codecs:
```
imageio-ffmpeg
av
pillow
```

Then adapt the writer in your code accordingly.

---

## 🔐 Privacy

Videos are processed in-memory/on-disk only for the duration of a session. If deploying publicly, consider adding:
- Upload size limits (via Streamlit Cloud settings)
- Auto-cleanup of temp folders after processing
- A simple disclaimer on data usage

---

## 📝 License

MIT (or your choice). Update this section to match your project’s license.
