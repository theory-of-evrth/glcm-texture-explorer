# GLCM Texture Explorer

An interactive Streamlit app for exploring Gray-Level Co-occurrence Matrices
(GLCMs) and common texture metrics. The app lets you choose bundled sample
images or upload your own image, converts it to grayscale, quantizes the gray
levels, computes the GLCM for a selected distance and direction, and displays
the resulting heatmap with texture metrics and a short value-based
interpretation.

## Features

- Built-in image samples plus custom image upload
- Adjustable number of gray levels
- Adjustable pixel distance and neighbor direction
- GLCM heatmap
- Texture metrics: contrast, homogeneity, energy, and correlation
- Dynamic interpretation text based on the computed metric values

## Local Run Instructions

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Hugging Face Space

```text
https://theory-of-evrth-glcm-texture-explorer-app-b79293.streamlit.app/
```

## Screenshot of the application

```markdown
![GLCM Texture Explorer demo](screenshots/screenshot.png)
```

## Known Limitations

- The interpretation thresholds are simple heuristics and are meant for
  teaching, not rigorous image classification.
- Metrics depend heavily on quantization level, distance, and direction, so
  comparisons are most meaningful when those settings stay fixed.
- Large uploaded images are resized before analysis to keep the app responsive.
