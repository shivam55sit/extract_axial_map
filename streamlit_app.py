import streamlit as st
import tempfile
import os
import cv2 as cv
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image

from axial_map_preprocessor import AxialMapProcessor


def to_png_bytes(cv_img: np.ndarray) -> bytes:
    # Convert BGR to PNG bytes
    success, buf = cv.imencode('.png', cv_img)
    if not success:
        raise RuntimeError('Failed to encode image to PNG')
    return buf.tobytes()


def bgr_to_pil(cv_img: np.ndarray) -> Image.Image:
    rgb = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def process_uploaded_image(uploaded_file) -> dict:
    # Save uploaded file to a temp path and run processor
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        processor = AxialMapProcessor()
        result = processor.process_image(tmp_path, output_dir=None)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return result


def main():
    st.title('Axial Map Extractor')
    st.write('Upload a Pentacam / Oculyzer image to extract and clean the axial curvature map.')

    uploaded = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg'])

    if uploaded is None:
        st.info('Awaiting image upload...')
        return

    st.image(uploaded, caption='Original image', use_column_width=True)

    if st.button('Process image'):
        with st.spinner('Processing...'):
            try:
                result = process_uploaded_image(uploaded)
            except Exception as e:
                st.error(f'Processing error: {e}')
                return

        if not result.get('success'):
            st.error(f"Failed: {result.get('message')}")
            return

        st.success('Processing successful')

        # Show metadata
        st.write('- Device:', result.get('device'))
        st.write('- Dimensions:', result.get('dimensions'))
        st.write('- Header:', result.get('header'))

        # Display full axial map
        axial_full = result.get('axial_map_full')
        axial_std = result.get('axial_map_224')

        if axial_full is not None:
            st.subheader('Axial map (full)')
            st.image(bgr_to_pil(axial_full), use_column_width=True)

            # Download button
            try:
                full_bytes = to_png_bytes(axial_full)
                st.download_button('Download full axial map', data=full_bytes,
                                   file_name='axial_full.png', mime='image/png')
            except Exception as e:
                st.warning(f'Could not prepare download: {e}')

        if axial_std is not None:
            st.subheader('Axial map (standardized 224x224)')
            st.image(bgr_to_pil(axial_std), use_column_width=False)

            try:
                std_bytes = to_png_bytes(axial_std)
                st.download_button('Download standardized axial map', data=std_bytes,
                                   file_name='axial_224.png', mime='image/png')
            except Exception as e:
                st.warning(f'Could not prepare download: {e}')


if __name__ == '__main__':
    main()
