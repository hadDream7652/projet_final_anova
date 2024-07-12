import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(
    page_title="ACCUEIL",
    page_icon="👋",
)

st.image('./files/logo_ensae.jpg', caption='Ecole nationale de la Statistique et de l\'Analyse économique', clamp=False, channels="RGB", output_format="auto")
st.snow()
st.sidebar.success("Présentation")
st.write("# **__Welcome to the app of DIOP Papa Boubacar and OUEDRAOGO Faïçal Cheick Hamed__**")
st.title('Projet d\'analyse de la variance')
st.markdown(
    """
    Cette application a été programmé dans le cadre d'un projet d'Analyse de la variance (ANOVA).
    Dans cette application, vous aurez la possibilité de mettre des bases sous format excel, ... pour faire de l'ANOVA.
"""
)
st.write('## Sous la direction de M. Pathé DIAKHATE')














