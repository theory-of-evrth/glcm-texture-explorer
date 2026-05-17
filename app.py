import streamlit as st
import matplotlib.pyplot as plt

from processing import load_image_as_gray, compute_glcm_features, scale_quantized_for_display


st.set_page_config(page_title="GLCM Texture Explorer", layout="wide")

st.title("GLCM Texture Explorer")
st.write(
    "Explore how Gray-Level Co-occurrence Matrices describe image texture "
    "by counting how often neighboring gray levels occur together."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
)

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

if uploaded_file is None:
    st.info("Upload an image to start.")
    st.stop()

try:
    gray_image = load_image_as_gray(uploaded_file)
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
        st.image(gray_image, clamp=True, width="stretch")

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

    st.write(
        "- Higher **contrast** usually means stronger local intensity variation.\n"
        "- Higher **homogeneity** means neighboring pixels are more similar.\n"
        "- Higher **energy** means the texture is more uniform or repetitive.\n"
        "- **Correlation** measures how predictable neighboring intensities are."
    )

except Exception as e:
    st.error("Something went wrong while processing the image.")
    st.exception(e)