import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats.multicomp import MultiComparison
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.libqsturng import psturng
st.set_page_config(page_title="Anova")
st.markdown("# Résultats de l'ANOVA")
st.sidebar.header("Résultats")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Hypothèses**", "**Résultats**", "**Interprétations**", "**Graphique**", "**Tests**"])

# Initialisation des variables de session
if 'facteur0' not in st.session_state:
    st.session_state.facteur0 = None
if 'facteur1' not in st.session_state:
    st.session_state.facteur1 = None
if 'facteur2' not in st.session_state:
    st.session_state.facteur2 = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'interaction' not in st.session_state:
    st.session_state.interaction = None
if 'anova_df' not in st.session_state:
    st.session_state.anova_df = pd.DataFrame()
if 'shapiro_test' not in st.session_state:
    st.session_state.shapiro_test = None
if 'fitted_vals' in st.session_state:
    st.session_state.fitted_vals = None
if 'levene_test' not in st.session_state:
    st.session_state.levene_test = None
if 'dw_test' not in st.session_state:
    st.session_state.dw_test = None
if 'shapiro_test' not in st.session_state:
    st.session_state.shapiro_test = None
if 'signfic' not in st.session_state:
    st.session_state.signific = None
with tab1:
    if 'data' in st.session_state and st.session_state.facteurs:
        tab1.write("### Vérification des hypothèses")

        st.session_state.signific = st.number_input("Veuillez choisir le seuil d'erreur pour les test d'hypothèses",min_value=0.01, max_value=0.1,step=0.01,value=0.01)

        if st.session_state.facteurs == 2 and st.session_state.data is not None and len(st.session_state.fac) != 0:
            st.session_state.interaction = tab1.selectbox("Type d'analyse", ['ANOVA avec interaction', 'ANOVA sans interaction'], index=None)
            tab1.write(st.session_state.interaction)

        if 'button_anova' not in st.session_state:
            st.session_state.button_anova = True

        if 'data' in st.session_state and len(st.session_state.fac) > 0 and not st.session_state.data.empty:
            if st.session_state.facteurs == 1:
                st.session_state.facteur0 = st.session_state.fac[0]
                formula = f"{st.session_state.variable_dependante} ~ C({st.session_state.facteur0})"
                if "variable_dependante" in st.session_state:
                    st.session_state.model = ols(formula, data=st.session_state.data).fit()
                    anova_lm(st.session_state.model, typ=2)

            if st.session_state.facteurs == 2:
                st.session_state.facteur1 = st.session_state.fac[0]
                st.session_state.facteur2 = st.session_state.fac[1]
                if st.session_state.interaction == "ANOVA sans interaction":
                    formula3 = f"{st.session_state.variable_dependante} ~ C({st.session_state.facteur2}) + C({st.session_state.facteur1})"
                    st.session_state.model = ols(formula3, data=st.session_state.data).fit()
                    anova_lm(st.session_state.model, typ=2)
                else:
                    formula2 = f"{st.session_state.variable_dependante} ~ C({st.session_state.facteur2}) * C({st.session_state.facteur1})"
                    st.session_state.model = ols(formula2, data=st.session_state.data).fit()
                    anova_lm(st.session_state.model, typ=2)

    if 'data' in st.session_state:
        with st.expander("**NORMALITE**"):
            st.header("Vérification de la normalité")
            # Histogramme des résidus
            st.write("**Histogramme des résidus**")
            fig, ax = plt.subplots()
            if st.session_state.model:
                sns.histplot(st.session_state.model.resid, kde=True, ax=ax)
            st.pyplot(fig)
            # Q-Q plot des résidus
            st.write("**Q-Q plot des résidus**")
            if st.session_state.model:    
                fig = sm.qqplot(st.session_state.model.resid, line='s')
                st.pyplot(fig)
            # Test de normalité de Shapiro-Wilk
            st.write("#### Test de Shapiro-Wilk")
            st.write("**Hypothèse nulle (H0)** : Les résidus suivent une distribution normale.")
            st.write("**Hypothèse alternative (H1)** : Les résidus ne suivent pas une distribution normale.")
            if st.session_state.model is not None:
                st.session_state.shapiro_test = stats.shapiro(st.session_state.model.resid)
                st.write(f"**Statistic**: {st.session_state.shapiro_test.statistic}, \n**P-value**: {st.session_state.shapiro_test.pvalue}")
                if st.session_state.shapiro_test.pvalue >= st.session_state.signific:
                    st.write("**L'hypothèse de normalité est vérifiée ✅**")
                else:
                    st.write("**L'hypothèse de normalité n'est pas vérifiée ❌**")

    if 'data' in st.session_state and st.session_state.data is not None:
        with st.expander("**HOMOSCEDASTICITE**"):
            st.header("Vérification de l'homogénéité")
            # Résidus vs valeurs ajustées
            st.write("**Graphique des résidus vs valeurs ajustées**")
            if 'model' in st.session_state and st.session_state.model is not None:
                st.session_state.fitted_vals = st.session_state.model.fittedvalues
                fig, ax = plt.subplots()
                ax.scatter(st.session_state.fitted_vals, st.session_state.model.resid)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel('Valeurs ajustées')
                ax.set_ylabel('Résidus')
                st.pyplot(fig)
            # Test de Levene pour l'homogénéité des variances
            st.write("#### Test de Levene")
            st.write("**Hypothèse nulle (H0)** : Les variances des groupes sont égales.")
            st.write("**Hypothèse alternative (H1)** : Les variances des groupes ne sont pas égales.")
            if st.session_state.facteurs == 1 and st.session_state.model is not None:
                st.session_state.levene_test = levene(*[group[st.session_state.variable_dependante].values for name, group in st.session_state.data.groupby(st.session_state.facteur0)])
            elif st.session_state.model is not None:
                st.session_state.levene_test = levene(*[group[st.session_state.variable_dependante].values for name, group in st.session_state.data.groupby([st.session_state.facteur1, st.session_state.facteur2])])
                st.write(f"**Statistic**: {st.session_state.levene_test.statistic}, \n**P-value**: {st.session_state.levene_test.pvalue}")
            if st.session_state.model is not None:    
                if st.session_state.levene_test.pvalue >= st.session_state.signific :
                    st.write("**L'hypothèse d'homogénéité est vérifiée ✅**")
                elif st.session_state.levene_test is not None:
                    st.write("**L'hypothèse d'homogénéité n'est pas vérifiée ❌**")

    if 'data' in st.session_state:
        with st.expander("**AUTOCORRELATION**"):
            st.header("Vérification de l'indépendance")
            st.write("#### Test de Durbin-Watson")
            st.write("**Hypothèse nulle (H0)** : Les résidus sont indépendants.")
            st.write("**Hypothèse alternative (H1)** : Les résidus ne sont pas indépendants.")
            # Test de Durbin-Watson pour l'indépendance des résidus
            if st.session_state.model is not None:   
                st.session_state.dw_test = durbin_watson(st.session_state.model.resid)
                st.session_state.dw_pvalue = 2 * (1 - stats.norm.cdf(abs(st.session_state.dw_test - 2)))
                st.write(f"**Statistic**: {st.session_state.dw_test}, \n**P-value**: {st.session_state.dw_pvalue}")
                if st.session_state.dw_pvalue >= st.session_state.signific:
                    st.write("**L'hypothèse d'indépendance est vérifiée ✅**")
                else:
                    st.write("**L'hypothèse d'indépendance n'est pas vérifiée ❌**")

# Significativité des p-value
def format_pvalues_with_stars(pvalues):
    formatted_pvalues = []
    stars = []
    for p in pvalues:
        if p < 0.01:
            stars.append('***')
        elif p < st.session_state.signific:
            stars.append('**')
        elif p < 0.1:
            stars.append('*')
        else:
            stars.append('')
        
        if p < 0.001:
            formatted_pvalues.append(f"{p:.2e}")
        else:
            formatted_pvalues.append(f"{p:.3f}")
    
    return formatted_pvalues, stars

if 'data' in st.session_state:
    if st.session_state.shapiro_test is not None and st.session_state.levene_test is not None:
        if st.session_state.shapiro_test.pvalue >= st.session_state.signific and st.session_state.levene_test.pvalue >= st.session_state.signific:
            with tab2:

                if st.session_state.facteurs == 2 and st.session_state.data is not None and len(st.session_state.fac) != 0:
                    tab2.write(st.session_state.interaction)
                if 'button_anova' not in st.session_state:
                    st.session_state.button_anova = False

                def click_button_anova():
                    st.session_state.button_anova = True
                
                st.button('Lancer l\'ANOVA', on_click=click_button_anova)
    
                if st.session_state.button_anova and 'data' in st.session_state and len(st.session_state.fac) > 0:
                    if st.session_state.facteurs == 1:
                        st.session_state.facteur0 = st.session_state.fac[0]
                        formula = f"{st.session_state.variable_dependante} ~ C({st.session_state.facteur0})"
                        st.session_state.model = ols(formula, data=st.session_state.data).fit()
                        st.session_state.anova_df = pd.DataFrame(anova_lm(st.session_state.model, typ=2))
                        st.session_state.anova_df['p-value'] = format_pvalues_with_stars(st.session_state.anova_df['PR(>F)'])[0]
                        st.session_state.anova_df['Significativité'] = format_pvalues_with_stars(st.session_state.anova_df['PR(>F)'])[1]


                    if st.session_state.facteurs == 2:
                        st.session_state.facteur1 = st.session_state.fac[0]
                        st.session_state.facteur2 = st.session_state.fac[1]
                        if st.session_state.interaction == "ANOVA sans interaction":
                            formula3 = f"{st.session_state.variable_dependante} ~ C({st.session_state.facteur2}) + C({st.session_state.facteur1})"
                            st.session_state.model = ols(formula3, data=st.session_state.data).fit()
                            st.session_state.anova_df = pd.DataFrame(anova_lm(st.session_state.model, typ=2))
                            st.session_state.anova_df['p-value'] = format_pvalues_with_stars(st.session_state.anova_df['PR(>F)'])[0]
                            st.session_state.anova_df['Significativité'] = format_pvalues_with_stars(st.session_state.anova_df['PR(>F)'])[1]
                        else:
                            formula2 = f"{st.session_state.variable_dependante} ~ C({st.session_state.facteur2}) * C({st.session_state.facteur1})"
                            st.session_state.model = ols(formula2, data=st.session_state.data).fit()
                            st.session_state.anova_df = pd.DataFrame(anova_lm(st.session_state.model, typ=2))
                            st.session_state.anova_df['p-value'] = format_pvalues_with_stars(st.session_state.anova_df['PR(>F)'])[0]
                            st.session_state.anova_df['Significativité'] = format_pvalues_with_stars(st.session_state.anova_df['PR(>F)'])[1]
                    st.write(st.session_state.anova_df)
                    # Effects by group
                    st.write("Effects by Group")
                    effects_summary = st.session_state.model.summary2().tables[1]
                    st.write(effects_summary)

                            
                
    # Onglet 3: Interprétations
                if 'data' in st.session_state and st.session_state.data is not None:
                    with tab3:
                        
                        subtab1, subtab2 = st.tabs(["Interprétations générales", "Interprétations détaillées"])

                        with subtab1:
                            subtab1.write("### Interprétations des résultats de l'ANOVA")
                            for index, row in st.session_state.anova_df.iterrows():
                                p_value = float(row['p-value'])
                                subtab1.write(f"**{index}**")
                                subtab1.write(f"**Statistique**: {row['F']}")
                                subtab1.write(f"**p-value**: {row['p-value']} **{row['Significativité']}**")
                                if p_value < st.session_state.signific:
                                    subtab1.write(f"Le facteur **{index}** a un effet significatif sur **{st.session_state.variable_dependante}**.")
                                else:
                                    subtab1.write(f"Le facteur **{index}** n'a pas d'effet significatif sur **{st.session_state.variable_dependante}**.")

                        with subtab2:
                            subtab2.write("### Interprétations détaillées")

                            # Création de la variable groupée
                            if st.session_state.facteur1 in st.session_state.data.columns and st.session_state.facteur2 in st.session_state.data.columns:
                                st.session_state.data['grp'] = st.session_state.data[st.session_state.facteur1].astype(str) + '-' + st.session_state.data[st.session_state.facteur2].astype(str)
                            # ANOVA a un facteur de la variable groupée
                            modele_grp = smf.ols(f'{st.session_state.variable_dependante} ~ grp', data=st.session_state.data).fit()
                            anova_results_df = pd.DataFrame(sm.stats.anova_lm(modele_grp, typ=2))
                            st.write("_Résultats de l'ANOVA à un facteur de la variable groupée_")
                            st.write(anova_results_df)
                            for index, row in anova_results_df.iterrows():
                                p_value = float(row['PR(>F)'])  # Conversion en float si nécessaire
                                subtab2.write(f"**{index}**")
                                subtab2.write(f"La statistique est: **{row['F']}**")
                                subtab2.write(f"La p-value est : **{p_value}**.")  # Affichez seulement la première valeur de p-value
                                if p_value < st.session_state.signific and row['F']>0:
                                    subtab2.write(f"Le facteur {index} a un effet significatif sur {st.session_state.variable_dependante}. Donc il y a une **synergie** entre {st.session_state.facteur1} et {st.session_state.facteur2}.")
                                elif p_value < st.session_state.signific and row['F']<0:
                                    subtab2.write(f"Le facteur {index} a un effet significatif sur {st.session_state.variable_dependante}. Donc, {st.session_state.facteur1} et {st.session_state.facteur2} sont **antagonistes**.")
                                else:
                                    subtab2.write(f"Le facteur {index} n'a pas d'effet significatif sur {st.session_state.variable_dependante}.")
                                break                 # ComparaisonMultiple Comparisons using Tukey's HSD
                            mc = MultiComparison(st.session_state.data[st.session_state.variable_dependante], st.session_state.data['grp'])
                            tukey_result = mc.tukeyhsd()
                            #st.write(tukey_result.summary())


                            # Visualization of interactions
                            #st.write("Interaction Plot")
                            #fig, ax = plt.subplots(figsize=(10, 6))
                            #interaction_plot(st.session_state.data[st.session_state.facteur1], st.session_state.data[st.session_state.facteur2], st.session_state.data[st.session_state.variable_dependante], ax=ax)
                            #st.pyplot(fig)

                    # Onglet 4: Graphiques
                    with tab4:

                        if st.session_state.facteurs == 1:
                            # Graphique pour un facteur
                            fig, ax = plt.subplots()
                            sns.boxplot(x=st.session_state.data[st.session_state.facteur0], y=st.session_state.data[st.session_state.variable_dependante], ax=ax)
                            ax.set_title(f"Boxplot de {st.session_state.variable_dependante} par {st.session_state.facteur0}")
                            st.pyplot(fig)
                        elif st.session_state.facteurs == 2:
                            if st.session_state.interaction == "ANOVA sans interaction":
                                # Graphique pour deux facteurs sans interaction
                                fig, ax = plt.subplots()
                                sns.boxplot(x=st.session_state.data[st.session_state.facteur1], y=st.session_state.data[st.session_state.variable_dependante], hue=st.session_state.data[st.session_state.facteur2], ax=ax)
                                ax.set_title(f"Boxplot de {st.session_state.variable_dependante} par {st.session_state.facteur1} et {st.session_state.facteur2}")
                                st.pyplot(fig)
                            else:
                                # Graphique pour deux facteurs avec interaction
                                fig, ax = plt.subplots()
                                sns.boxplot(x=st.session_state.data[st.session_state.facteur1], y=st.session_state.data[st.session_state.variable_dependante], hue=st.session_state.data[st.session_state.facteur2], ax=ax)
                                ax.set_title(f"Boxplot de {st.session_state.variable_dependante} par {st.session_state.facteur1} et {st.session_state.facteur2} avec interaction")
                                st.pyplot(fig)

                            st.write("Graphique de Tukey")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            tukey_result.plot_simultaneous(ax=ax)
                            st.pyplot(fig)

            with tab5:


                    # Comparisons significance with letters
                    st.write("Comparison of Significance with Letters")
        #tukey_groups = mc.groupsunique
        #letters = tukey_result._results_table.data[1:]

        #letters_df = pd.DataFrame({'grp': tukey_groups, 'letters': [x[2] for x in letters]})
        #st.write(letters_df)

        # Boxplot with Tukey's HSD letters
        #st.write("Boxplot with Tukey's HSD Letters")
        #fig, ax = plt.subplots(figsize=(10, 6))
        #sns.boxplot(x='grp', y=st.session_state.variable_dependante, data=st.session_state.data, ax=ax)
        #sns.swarmplot(x='grp', y=st.session_state.variable_dependante, data=st.session_state.data, color=".25", ax=ax)

        # Adding Tukey's HSD letters to the plot
        #for i, grp in enumerate(tukey_groups):
        #    ax.text(i, st.session_state.data[st.session_state.variable_dependante].max(), letters_df.loc[letters_df['grp'] == grp, 'letters'].values[0], horizontalalignment='center', size='medium', color='black')

        #st.pyplot(fig)
            # Graphique des différences de moyennes avec intervalles de confiance
        #tukey = pairwise_tukeyhsd(st.session_state.data[st.session_state.variable_dependante], st.session_state.data[st.session_state.facteur0])
        #fig, ax = plt.subplots()
        #tukey.plot_simultaneous(ax=ax)
        #ax.set_title(f"Différences de moyennes avec intervalles de confiance pour {st.session_state.facteur0}")
        #st.pyplot(fig)

        # Graphique des différences de moyennes avec intervalles de confiance
        #tukey1 = pairwise_tukeyhsd(st.session_state.data[st.session_state.variable_dependante], st.session_state.data[st.session_state.facteur1])
        #tukey2 = pairwise_tukeyhsd(st.session_state.data[st.session_state.variable_dependante], st.session_state.data[st.session_state.facteur2])
        #fig, ax = plt.subplots()
        #tukey1.plot_simultaneous(ax=ax)
        #ax.set_title(f"Différences de moyennes avec intervalles de confiance pour {st.session_state.facteur1}")
        #st.pyplot(fig)
        #fig, ax = plt.subplots()
        #tukey2.plot_simultaneous(ax=ax)
        #ax.set_title(f"Différences de moyennes avec intervalles de confiance pour {st.session_state.facteur2}")
        #st.pyplot(fig)

        # Graphique des différences de moyennes avec intervalles de confiance pour l'interaction
        #tukey_interaction = pairwise_tukeyhsd(st.session_state.data[st.session_state.variable_dependante], st.session_state.data[st.session_state.facteur1].astype(str) + "-" + st.session_state.data[st.session_state.facteur2].astype(str))
        #fig, ax = plt.subplots()
        #tukey_interaction.plot_simultaneous(ax=ax)
        #ax.set_title(f"Différences de moyennes avec intervalles de confiance pour l'interaction {st.session_state.facteur1} et {st.session_state.facteur2}")
        #st.pyplot(fig)
        else:
            st.write("Les hypothèses de l'ANOVA ne sont pas vérifiées donc nous suggerons une analyse non paramétrique")
