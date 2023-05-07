"""
Main script: drives the logic and basic structure of
the Streamlit web application
"""

import pages
import streamlit as st


def run() -> None:
    """
    Sets the background and a navigation bar to access the homepage
    as well as register and login pages.
    """
    pages.set_icon_image()
    page_names_to_funcs = {
        "Movie Recommender": pages.homepage,
        "Login": pages.login_page,
        "Register": pages.register_page,
        "Profile Page": pages.profile_page,
        "Information on the dataset": pages.data_page,}

    demo_name = st.sidebar.selectbox("Navigate", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()


if __name__ == '__main__':
    run()
