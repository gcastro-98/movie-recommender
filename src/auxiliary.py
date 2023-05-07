"""
Implementation of some auxiliary routines.
"""

import streamlit as st
import base64


@st.cache(allow_output_mutation=True)
def get_base64_image(image_path: str) -> str:
    """
    Decode a .png image into a str for the streamlit to be able to use it.
    Uses the streamlit caching to stay performant even when loading.

    Parameters
    ----------
    image_path: str
        Local path to the image

    Returns
    -------
    repr: str
        String representation of the image
    """
    with open(image_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
