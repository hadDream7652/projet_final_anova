import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import configu
import seaborn as sns


configu.interaction = True


st.set_page_config(page_title="Données", page_icon="📈")
st.markdown("# Base à analyser")
st.sidebar.header("Données")
tab1, tab2, tab3 = st.tabs(["Base", "Statistiques", "Boxplot"],)

uploaded = tab1.file_uploader("Choisissez la base")

if uploaded is not None and uploaded != st.session_state.get('uploaded', None):
    st.session_state.clear()
    st.session_state.uploaded = uploaded
    st.session_state.data = None
else:
    uploaded = st.session_state.get('uploaded', None)
if st.button("Charger la base par défaut"):
    st.session_state.clear()
    st.session_state.uploaded = "https://raw.githubusercontent.com/hadDream7652/test_anova2/main/projet_final3/files/Base_test.dta"
    st.session_state.data = pd.read_stata(st.session_state.uploaded)


if 'data' not in st.session_state:
    st.session_state.data = None

if 'uploaded' not in st.session_state:
    st.session_state.uploaded = None

with tab1:
    st.header('Vue de la base')
    if uploaded is not None:
        st.session_state.uploaded = uploaded
        try:
            st.session_state.data = pd.read_excel(uploaded)
        except:
            try:
                st.session_state.data = pd.read_stata(uploaded)
            except:
                try:
                    st.session_state.data = pd.read_csv(uploaded)
                except:
                    st.session_state.data = pd.read_sas(uploaded)

    if st.session_state.data is not None:
        tab1.write(st.session_state.data)

    if 'fac' not in st.session_state:
        st.session_state.fac = []

    if 'variable_dependante' not in st.session_state:
        st.session_state.variable_dependante = None
    if st.session_state.data is not None:
        st.session_state.variable_dependante = tab1.selectbox("Sélectionner la variable étudiée", st.session_state.data.select_dtypes(np.number).columns, index=st.session_state.data.select_dtypes(np.number).columns.get_loc(st.session_state.variable_dependante) if st.session_state.variable_dependante is not None else 0)
        configu.variable_dependante = st.session_state.variable_dependante
    configu.data = st.session_state.data

    if 'facteurs' not in st.session_state:
        st.session_state.facteurs = 1

    if st.session_state.data is not None:
        if len(configu.data.select_dtypes(include=['object','category']).columns) == 1:
            st.session_state.facteurs = 1
        elif len(configu.data.select_dtypes(include=['object','category']).columns) >= 2:
            tab1.options = tab1.selectbox("Quel ANOVA voulez-vous faire?", ["1 facteur", "2 facteurs"], index=st.session_state.facteurs-1)
            st.session_state.facteurs = int(tab1.options.split()[0])
    configu.facteurs = st.session_state.facteurs

    if 'potentiels_facteurs' not in st.session_state:
        st.session_state.potentiels_facteurs = []

    if st.session_state.uploaded is not None:
        st.session_state.potentiels_facteurs = [col for col in configu.data.select_dtypes(include=['object','category']).columns if configu.data[col].nunique() > 1]
        configu.potentiels_facteurs = st.session_state.potentiels_facteurs



    if st.session_state.uploaded is not None and len(configu.potentiels_facteurs) == 1:
        configu.fac = configu.potentiels_facteurs
    elif st.session_state.uploaded is not None and len(configu.potentiels_facteurs) == 0:
        tab1.markdown("Veuillez choisir une base où l'une des variables peut être un facteur")
    elif st.session_state.uploaded is not None and configu.facteurs == 1 and list(configu.data.select_dtypes(include=['object','category']).columns) == configu.potentiels_facteurs:
        st.session_state.fac = [tab1.selectbox(
            "Veuillez choisir un facteur",
            st.session_state.potentiels_facteurs,
            index=(st.session_state.potentiels_facteurs.index(st.session_state.fac[0]) if st.session_state.fac else 0)
        )]
        configu.fac = st.session_state.fac
    elif st.session_state.uploaded is not None and configu.facteurs == 2:
        st.session_state.fac = tab1.multiselect("Veuillez choisir les facteurs",st.session_state.potentiels_facteurs,max_selections=2, default=st.session_state.fac if len(st.session_state.fac if st.session_state.fac else [1])==2 else None)       
        configu.fac = tab1.options
with tab2:

    st.header(" Statistiques descriptives")
    if st.session_state.data is not None and len(st.session_state.fac) != 0 and st.session_state.variable_dependante is not None:
        try:
            tab2.write(st.session_state.data.groupby(st.session_state.fac)[st.session_state.variable_dependante].mean())
        except:
            tab2.write()
        for i in st.session_state.fac:
            st.session_state.data[i] = st.session_state.data[i].astype('category')
        configu.data = st.session_state.data

    if st.session_state.data is not None:
        variables_quantitatives = st.session_state.data.select_dtypes(include=[np.number])
        tab2.write(st.session_state.data.select_dtypes(include=[np.number]).describe().transpose())





with tab3:
    # Check if data and variables are selected
    if st.session_state.data is not None and st.session_state.variable_dependante is not None and st.session_state.facteurs == 2:

        # Define custom colors
        st.header("Boxplot de la variable à étudier")
        unique_varietes = st.session_state.data[st.session_state.fac[1]].unique() if len(st.session_state.fac if st.session_state.fac else [0])>1 else None
        colors = ['blue', 'orange', 'white', 'red'][:len(unique_varietes)] if st.session_state.fac and unique_varietes is not None else None
        palette = dict(zip(unique_varietes, colors)) if st.session_state.fac and unique_varietes is not None else None
        # Create the plot
        plt.figure(figsize=(10, 6))
            # Boxplot
        
