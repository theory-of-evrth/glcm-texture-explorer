import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

from processing import load_image_as_gray, compute_glcm_features, scale_quantized_for_display


st.set_page_config(page_title="GLCM Texture Explorer", layout="wide")

st.title("GLCM Texture Explorer")
st.write(
    "Explore how Gray-Level Co-occurrence Matrices describe image texture "
    "by counting how often neighboring gray levels occur together."
)

SAMPLE_IMAGE_DIR = Path(__file__).parent / "image samples"
SAMPLE_IMAGE_TYPES = {".png", ".jpg", ".jpeg"}


def find_sample_images():
    if not SAMPLE_IMAGE_DIR.exists():
        return []

    return sorted(
        path
        for path in SAMPLE_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SAMPLE_IMAGE_TYPES
    )


def describe_texture(features):
    contrast = features["contrast"]
    homogeneity = features["homogeneity"]
    energy = features["energy"]
    correlation = features["correlation"]

    if contrast < 2:
        contrast_text = "The low contrast suggests neighboring pixels usually have similar brightness, so the texture is fairly smooth."
    elif contrast < 10:
        contrast_text = "The moderate contrast points to visible local intensity changes without an extremely sharp or noisy texture."
    else:
        contrast_text = "The high contrast suggests strong local brightness changes, which often appear as edges, grain, or abrupt texture transitions."

    if homogeneity > 0.7:
        homogeneity_text = "High homogeneity reinforces that nearby pixels often fall into similar gray levels."
    elif homogeneity > 0.4:
        homogeneity_text = "The medium homogeneity suggests a mix of smooth regions and local variation."
    else:
        homogeneity_text = "Low homogeneity means neighboring gray levels are often different, so the texture is less locally uniform."

    if energy > 0.3:
        energy_text = "The high energy indicates the GLCM is concentrated in a few patterns, so the image has repeated or uniform structure."
    elif energy > 0.15:
        energy_text = "The energy is moderate, meaning the texture has some repeated structure but also a spread of gray-level pairings."
    else:
        energy_text = "The low energy shows the gray-level pairings are widely spread, which usually means a more varied texture."

    if correlation > 0.75:
        correlation_text = "Strong correlation means neighboring intensities are quite predictable from one another."
    elif correlation > 0.35:
        correlation_text = "Moderate correlation means neighboring intensities have some relationship, but the pattern is not rigid."
    else:
        correlation_text = "Weak correlation means neighboring intensities are less predictable in this direction and distance."

    return [contrast_text, homogeneity_text, energy_text, correlation_text]


sample_images = find_sample_images()

st.sidebar.header("Image")
source_options = []
if sample_images:
    source_options.append("Sample image")
source_options.append("Upload image")

image_source = st.sidebar.radio("Source", source_options, horizontal=True)

selected_image = None
selected_image_name = None

if image_source == "Sample image":
    sample_lookup = {path.stem.replace("-", " ").replace("_", " "): path for path in sample_images}
    selected_image_name = st.sidebar.selectbox("Sample", options=list(sample_lookup.keys()))
    selected_image = sample_lookup[selected_image_name]
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        selected_image = uploaded_file
        selected_image_name = uploaded_file.name

st.sidebar.header("GLCM parameters")

levels = st.sidebar.slider(
    "Number of gray levels",
    min_value=4,
    max_value=64,
    value=16,
    step=4,
    help="Compresses the image brightness values into fewer levels. Lower values make the GLCM simpler; higher values preserve more detail but can make the matrix noisier."
)

distance = st.sidebar.slider(
    "Pixel distance",
    min_value=1,
    max_value=20,
    value=1,
    help="Controls how far apart the compared pixel pairs are. Distance 1 compares immediate neighbors; larger distances capture broader texture patterns."
)

angle = st.sidebar.selectbox(
    "Neighbor direction",
    options=[0, 45, 90, 135],
    index=0,
    help="Controls the direction of the neighboring pixel. 0° checks horizontal neighbors, 90° vertical neighbors, and 45°/135° diagonal neighbors."
)

if selected_image is None:
    st.info("Choose a sample image or upload an image to start.")
    st.stop()

try:
    gray_image = load_image_as_gray(selected_image)
    quantized, glcm_matrix, features = compute_glcm_features(
        gray_image,
        levels=levels,
        distance=distance,
        angle_degrees=angle,
    )

    quantized_display = scale_quantized_for_display(quantized, levels)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input grayscale image")
        st.image(gray_image, clamp=True, width="stretch", caption=selected_image_name)

    with col2:
        st.subheader("Quantized image")
        st.image(quantized_display, clamp=True, width="stretch")

    st.subheader("GLCM heatmap")

    fig, ax = plt.subplots()
    im = ax.imshow(glcm_matrix)
    ax.set_xlabel("Neighbor gray level")
    ax.set_ylabel("Reference gray level")
    ax.set_title("Gray-Level Co-occurrence Matrix")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

    st.subheader("Texture metrics")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Contrast", f"{features['contrast']:.3f}")
    m2.metric("Homogeneity", f"{features['homogeneity']:.3f}")
    m3.metric("Energy", f"{features['energy']:.3f}")
    m4.metric("Correlation", f"{features['correlation']:.3f}")

    st.subheader("Interpretation")

    st.write(" ".join(describe_texture(features)))

except Exception as e:
    st.error("Something went wrong while processing the image.")
    st.exception(e)
