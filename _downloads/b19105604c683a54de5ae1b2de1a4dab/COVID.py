# -*- coding: utf-8 -*-
"""

.. _ex_covid:


Example: Predicting COVID-19 patients outcome using MRCs
==================================================================

In this example we will use `MRCpy.MRC` and `MRCpy.CMRC` to predict the outcome
of a COVID-19 positive patient at the moment of hospital triage. This example
uses a dataset that comprises different demographic variables and biomarkers of
the patients and a binary outcome :attr:`Status` where :attr:`Status = 0`
define the group of survivors and :attr:`Status = 1` determines a decease.

The data is provided by the `Covid Data Saves Lives
<https://www.hmhospitales.com/coronavirus/covid-data-save-lives/>`_ initiative
carried out by HM Hospitales with information of the first wave of the COVID
outbreak in Spanish hospitals. The data is available upon request through HM
Hospitales
`here <https://www.hmhospitales.com/coronavirus/covid-data-save-lives/>`_ .

.. seealso::    For more information about the dataset and the creation of a
                risk indicator using Logistic regression refer to:

                [1] Ruben Armañanzas et al. “Derivation of a Cost-Sensitive
                COVID-19 Mortality Risk Indicator Using a Multistart Framework"
                , in *2021 IEEE International Conference on Bioinformatics and
                Biomedicine (BIBM)*, 2021, pp. 2179–2186.



First we will see how to deal with class imbalance when training a model using
syntethic minority over-sampling (SMOTE) techniques. Furthermore, we will
comparetwo MRC with two state of the art machine learning models probability
estimation . The selected models are :mod:`CMRC(phi = 'threshold' ,
loss = 'log')` & :mod:`MRC(phi = 'fourier' , loss = 'log')` for  the group of
MRCs and Logistic Regression (LR) & C-Support Vector Classifier(SVC) with the
implementation from `Scikit-Learn <https://scikit-learn.org/stable/#>`.


"""

# Import needed modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from MRCpy import CMRC, MRC


#############################################
# COVID dataset Loader:
# --------------------------------


def load_covid(norm=False, array=True):
    data_consensus = pd.read_csv("data/data_consensus.csv", sep=";")
    # rename variables
    variable_dict = {
        "CD0000AGE": "Age",
        "CORE": "PATIENT_ID",
        "CT000000U": "Urea",
        "CT00000BT": "Bilirubin",
        "CT00000NA": "Sodium",
        "CT00000TP": "Proth_time",
        "CT0000COM": "Com",
        "CT0000LDH": "LDH",
        "CT0000NEU": "Neutrophils",
        "CT0000PCR": "Pro_C_Rea",
        "CT0000VCM": "Med_corp_vol",
        "CT000APTT": "Ceph_time",
        "CT000CHCM": "Mean_corp_Hgb",
        "CT000EOSP": "Eosinophils%",
        "CT000LEUC": "Leukocytes",
        "CT000LINP": "Lymphocytes%",
        "CT000NEUP": "Neutrophils%",
        "CT000PLAQ": "Platelet_count",
        "CTHSDXXRATE": "Rate",
        "CTHSDXXSAT": "Sat",
        "ED0DISWHY": "Status",
        "F_INGRESO/ADMISSION_D_ING/INPAT": "Fecha_admision",
        "SEXO/SEX": "Sexo",
    }
    data_consensus = data_consensus.rename(columns=variable_dict)
    if norm:  # if we want the data standardised
        x_consensus = data_consensus[
            data_consensus.columns.difference(["Status", "PATIENT_ID"])
        ][:]
        std_scale = preprocessing.StandardScaler().fit(x_consensus)
        x_consensus_std = std_scale.transform(x_consensus)
        dataframex_consensus = pd.DataFrame(
            x_consensus_std, columns=x_consensus.columns
        )
        data_consensus.reset_index(drop=True, inplace=True)
        data_consensus = pd.concat(
            [dataframex_consensus, data_consensus[["Status"]]], axis=1
        )
    data_consensus = data_consensus[
        data_consensus.columns.difference(["PATIENT_ID"])
    ]
    X = data_consensus[
        data_consensus.columns.difference(["Status", "PATIENT_ID"])
    ]
    y = data_consensus["Status"]
    if array:
        X = X.to_numpy()
        y = y.to_numpy()
    return X, y


#############################################
# Addressing dataset imbalance with SMOTE
# --------------------------------
# The COVID dataset has a significant problem of class imbalance where the
# positive outcome has a prevalence of 85% (1522) whilst the negative outcome
# has only 276. In this example oversampling will be used to add syintetic
# records to get an almost balanced dataset. :mod:`SMOTE` (Synthetic minority
# over sampling) is a package that implements such oversampling.
X, y = load_covid(array=False)
described = (
    X.describe(percentiles=[0.5])
    .round(2)
    .transpose()[["count", "mean", "std"]]
)
pd.DataFrame(y.value_counts().rename({0.0: "Survive", 1.0: "Decease"}))


##############################################
# .. raw:: html
#
#     <div class="output_subarea output_html rendered_html output_result">
#     <div>
#     <style scoped>
#         .dataframe tbody tr th:only-of-type {
#             vertical-align: middle;
#         }
#
#         .dataframe tbody tr th {
#             vertical-align: top;
#         }
#
#         .dataframe thead th {
#             text-align: right;
#         }
#     </style>
#     <table border="1" class="dataframe">
#       <thead>
#         <tr style="text-align: right;">
#           <th></th>
#           <th>Status</th>
#         </tr>
#       </thead>
#       <tbody>
#         <tr>
#           <th>Survive</th>
#           <td>1522</td>
#         </tr>
#         <tr>
#           <th>Decease</th>
#           <td>276</td>
#         </tr>
#       </tbody>
#     </table>
#     </div>
#     </div>
#     <br />
#     <br />
##############################################

##############################################
# So we create a set of cases syntehtically using 5 nearest neighbors until
# the class imbalance is almost removed. For more information about
# :mod:`SMOTE` refer to it's `documentation
# <https://imbalanced-learn.org/stable/>`_ .
# We will use the method `SMOTE-NC` for numerical and categorical variables.
#
# .. seealso::    For more information about the SMOTE package refer to:
#
#                [2] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer,
#                W. P. (2002). SMOTE: synthetic minority over-sampling
#                technique. Journal of artificial intelligence research,
#                16, 321-357.

# We fit the data to the oversampler
smotefit = SMOTENC(sampling_strategy=0.75, categorical_features=[3])
X_resampled, y_resampled = smotefit.fit_resample(X, y)
described_resample = (
    X_resampled.describe(percentiles=[0.5])
    .round(2)
    .transpose()[["count", "mean", "std"]]
)
described_resample = described_resample.add_suffix("_SMT")
pd.concat([described, described_resample], axis=1)

##############################################
# .. raw:: html
#
#     <div class="output_subarea output_html rendered_html output_result">
#     <div>
#     <style scoped>
#         .dataframe tbody tr th:only-of-type {
#             vertical-align: middle;
#         }
#
#         .dataframe tbody tr th {
#             vertical-align: top;
#         }
#
#         .dataframe thead th {
#             text-align: right;
#         }
#     </style>
#     <table border="1" class="dataframe">
#       <thead>
#         <tr style="text-align: right;">
#           <th></th>
#           <th>count</th>
#           <th>mean</th>
#           <th>std</th>
#           <th>count_SMT</th>
#           <th>mean_SMT</th>
#           <th>std_SMT</th>
#         </tr>
#       </thead>
#       <tbody>
#         <tr>
#           <th>Age</th>
#           <td>1798.0</td>
#           <td>67.79</td>
#           <td>15.67</td>
#           <td>2663.0</td>
#           <td>71.71</td>
#           <td>14.78</td>
#         </tr>
#         <tr>
#           <th>Bilirubin</th>
#           <td>1798.0</td>
#           <td>0.57</td>
#           <td>0.45</td>
#           <td>2663.0</td>
#           <td>0.60</td>
#           <td>0.49</td>
#         </tr>
#         <tr>
#           <th>Ceph_time</th>
#           <td>1798.0</td>
#           <td>32.94</td>
#           <td>7.03</td>
#           <td>2663.0</td>
#           <td>33.32</td>
#           <td>7.50</td>
#         </tr>
#         <tr>
#           <th>Com</th>
#           <td>1798.0</td>
#           <td>0.50</td>
#           <td>0.78</td>
#           <td>2663.0</td>
#           <td>0.49</td>
#           <td>0.78</td>
#         </tr>
#         <tr>
#           <th>Eosinophils%</th>
#           <td>1798.0</td>
#           <td>0.70</td>
#           <td>1.57</td>
#           <td>2663.0</td>
#           <td>0.55</td>
#           <td>1.33</td>
#         </tr>
#         <tr>
#           <th>LDH</th>
#           <td>1798.0</td>
#           <td>601.10</td>
#           <td>367.24</td>
#           <td>2663.0</td>
#           <td>675.02</td>
#           <td>471.53</td>
#         </tr>
#         <tr>
#           <th>Leukocytes</th>
#           <td>1798.0</td>
#           <td>7.62</td>
#           <td>4.54</td>
#           <td>2663.0</td>
#           <td>8.23</td>
#           <td>4.86</td>
#         </tr>
#         <tr>
#           <th>Lymphocytes%</th>
#           <td>1798.0</td>
#           <td>18.19</td>
#           <td>10.44</td>
#           <td>2663.0</td>
#           <td>16.24</td>
#           <td>9.92</td>
#         </tr>
#         <tr>
#           <th>Mean_corp_Hgb</th>
#           <td>1798.0</td>
#           <td>33.62</td>
#           <td>1.42</td>
#           <td>2663.0</td>
#           <td>33.52</td>
#           <td>1.35</td>
#         </tr>
#         <tr>
#           <th>Med_corp_vol</th>
#           <td>1798.0</td>
#           <td>88.23</td>
#           <td>5.77</td>
#           <td>2663.0</td>
#           <td>88.63</td>
#           <td>5.88</td>
#         </tr>
#         <tr>
#           <th>Neutrophils</th>
#           <td>1798.0</td>
#           <td>5.75</td>
#           <td>3.77</td>
#           <td>2663.0</td>
#           <td>6.44</td>
#           <td>4.09</td>
#         </tr>
#         <tr>
#           <th>Neutrophils%</th>
#           <td>1798.0</td>
#           <td>73.01</td>
#           <td>12.99</td>
#           <td>2663.0</td>
#           <td>75.54</td>
#           <td>12.56</td>
#         </tr>
#         <tr>
#           <th>Platelet_count</th>
#           <td>1798.0</td>
#           <td>225.32</td>
#           <td>96.93</td>
#           <td>2663.0</td>
#           <td>219.27</td>
#           <td>93.65</td>
#         </tr>
#         <tr>
#           <th>Pro_C_Rea</th>
#           <td>1798.0</td>
#           <td>101.00</td>
#           <td>100.87</td>
#           <td>2663.0</td>
#           <td>121.41</td>
#           <td>110.35</td>
#         </tr>
#         <tr>
#           <th>Proth_time</th>
#           <td>1798.0</td>
#           <td>15.39</td>
#           <td>13.89</td>
#           <td>2663.0</td>
#           <td>16.17</td>
#           <td>15.14</td>
#         </tr>
#         <tr>
#           <th>Rate</th>
#           <td>1798.0</td>
#           <td>79.29</td>
#           <td>14.75</td>
#           <td>2663.0</td>
#           <td>80.69</td>
#           <td>14.81</td>
#         </tr>
#         <tr>
#           <th>Sat</th>
#           <td>1798.0</td>
#           <td>94.67</td>
#           <td>4.81</td>
#           <td>2663.0</td>
#           <td>93.60</td>
#           <td>5.96</td>
#         </tr>
#         <tr>
#           <th>Sodium</th>
#           <td>1798.0</td>
#           <td>136.92</td>
#           <td>4.50</td>
#           <td>2663.0</td>
#           <td>137.21</td>
#           <td>4.93</td>
#         </tr>
#         <tr>
#           <th>Urea</th>
#           <td>1798.0</td>
#           <td>43.17</td>
#           <td>30.72</td>
#           <td>2663.0</td>
#           <td>49.75</td>
#           <td>32.74</td>
#         </tr>
#       </tbody>
#     </table>
#     </div>
#     </div>
#     <br />
#     <br />
##############################################
##############################################
# We see how the distribution of the real data and the resampled data is
# different. However the distribution between classes is kept similar due to
# the creation of the synthetic cases through 5 nearest neighbors.

pd.DataFrame(
    y_resampled.value_counts().rename({0.0: "Survive", 1.0: "Decease"})
)


##############################################
# .. raw:: html
#
#     <div class="output_subarea output_html rendered_html output_result">
#     <div>
#     <style scoped>
#         .dataframe tbody tr th:only-of-type {
#             vertical-align: middle;
#         }
#
#         .dataframe tbody tr th {
#             vertical-align: top;
#         }
#
#         .dataframe thead th {
#             text-align: right;
#         }
#     </style>
#     <table border="1" class="dataframe">
#       <thead>
#         <tr style="text-align: right;">
#           <th></th>
#           <th>Status</th>
#         </tr>
#       </thead>
#       <tbody>
#         <tr>
#           <th>Survive</th>
#           <td>1522</td>
#         </tr>
#         <tr>
#           <th>Decease</th>
#           <td>1141</td>
#         </tr>
#       </tbody>
#     </table>
#     </div>
#     </div>
#     <br />
#     <br />

##############################################


#############################################
# Probability estimation
# ----------------------------------
# In this section we will estimate the conditional probabilities and analyse
# the distribution of the probabilities depending on the real outcome . The
# probability estimation is better when using :mod:`loss = log`. We use
# :mod:`CMRC(phi = 'threshold', loss = 'log')` and
# :mod:`MRC(phi = 'fourier' , loss = 'log'`. We will then compare these MRCs
# with SVC and LR with default parameters.

#############################################
# Load classification function:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# These function classify each of the cases in their correspondent
# confusion matrix's category. It also allows to set the desired cut-off
# for the predictions.


def defDataFrame(model, x_test, y_test, threshold=0.5):
    """
    Takes x,y test and train and a fitted model and
    computes the probabilities to then classify in TP,TN , FP , FN.
    """
    if "predict_proba" in dir(model):
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = [1 if i > threshold else 0 for i in probabilities]
        df = pd.DataFrame(
            {
                "Real": y_test.tolist(),
                "Prediction": predictions,
                "Probabilities": probabilities.tolist(),
            }
        )
    else:
        df = pd.DataFrame(
            {"Real": y_test.tolist(), "Prediction": model.predict(x_test)}
        )
    conditions = [
        (df["Real"] == 1) & (df["Prediction"] == 1),
        (df["Real"] == 1) & (df["Prediction"] == 0),
        (df["Real"] == 0) & (df["Prediction"] == 0),
        (df["Real"] == 0) & (df["Prediction"] == 1),
    ]
    choices = [
        "True Positive",
        "False Negative",
        "True Negative",
        "False Positive",
    ]
    df["Category"] = np.select(conditions, choices, default="No")
    df.sort_index(inplace=True)
    df.sort_values(by="Category", ascending=False, inplace=True)
    return df


#############################################
# Train models:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We will train the models with 80% of the data and then test with the other
# 20% selected randomly.

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=1
)

clf_MRC = MRC(phi="fourier", use_cvx=True, loss="log").fit(X_train, y_train)
df_MRC = defDataFrame(model=clf_MRC, x_test=X_test, y_test=y_test)
MRC_values = pd.DataFrame(df_MRC.Category.value_counts()).rename(
    columns={"Category": type(clf_MRC).__name__}
)
MRC_values["Freq_MRC"] = MRC_values["MRC"] / sum(MRC_values["MRC"]) * 100

clf_CMRC = CMRC(phi="threshold", use_cvx=True, loss="log").fit(
    X_train, y_train
)
df_CMRC = defDataFrame(model=clf_CMRC, x_test=X_test, y_test=y_test)
CMRC_values = pd.DataFrame(df_CMRC.Category.value_counts()).rename(
    columns={"Category": type(clf_CMRC).__name__}
)
CMRC_values["Freq_CMRC"] = CMRC_values["CMRC"] / sum(CMRC_values["CMRC"]) * 100

clf_SVC = SVC(probability=True).fit(X_train, y_train)
df_SVC = defDataFrame(model=clf_SVC, x_test=X_test, y_test=y_test)
SVC_values = pd.DataFrame(df_SVC.Category.value_counts()).rename(
    columns={"Category": type(clf_SVC).__name__}
)
SVC_values["Freq_SVC"] = SVC_values["SVC"] / sum(SVC_values["SVC"]) * 100

clf_LR = LogisticRegression().fit(X_train, y_train)
df_LR = defDataFrame(model=clf_LR, x_test=X_test, y_test=y_test)
LR_values = pd.DataFrame(df_LR.Category.value_counts()).rename(
    columns={"Category": type(clf_LR).__name__}
)
LR_values["Freq_LR"] = (
    LR_values["LogisticRegression"]
    / sum(LR_values["LogisticRegression"])
    * 100
)


pd.concat(
    [MRC_values, CMRC_values, SVC_values, LR_values], axis=1
).style.set_caption("Classification results by model").format(precision=2)

################################################
# .. raw:: html
#
#     <div class="output_subarea output_html rendered_html output_result">
#     <style type="text/css">
#     </style>
#     <table id="T_2cf59_">
#       <caption>Classification results by model</caption>
#       <thead>
#         <tr>
#           <th class="blank level0" >&nbsp;</th>
#           <th class="col_heading level0 col0" >MRC</th>
#           <th class="col_heading level0 col1" >Freq_MRC</th>
#           <th class="col_heading level0 col2" >CMRC</th>
#           <th class="col_heading level0 col3" >Freq_CMRC</th>
#           <th class="col_heading level0 col4" >SVC</th>
#           <th class="col_heading level0 col5" >Freq_SVC</th>
#           <th class="col_heading level0 col6" >LogisticRegression</th>
#           <th class="col_heading level0 col7" >Freq_LR</th>
#         </tr>
#       </thead>
#       <tbody>
#         <tr>
#           <th id="T_2cf59_level0_row0" class="row_heading
#           level0 row0" >True Negative</th>
#           <td id="T_2cf59_row0_col0" class="data row0 col0" >278</td>
#           <td id="T_2cf59_row0_col1" class="data row0 col1" >52.16</td>
#           <td id="T_2cf59_row0_col2" class="data row0 col2" >279</td>
#           <td id="T_2cf59_row0_col3" class="data row0 col3" >52.35</td>
#           <td id="T_2cf59_row0_col4" class="data row0 col4" >249</td>
#           <td id="T_2cf59_row0_col5" class="data row0 col5" >46.72</td>
#           <td id="T_2cf59_row0_col6" class="data row0 col6" >267</td>
#           <td id="T_2cf59_row0_col7" class="data row0 col7" >50.09</td>
#         </tr>
#         <tr>
#           <th id="T_2cf59_level0_row1" class="row_heading
#           level0 row1" >False Negative</th>
#           <td id="T_2cf59_row1_col0" class="data row1 col0" >140</td>
#           <td id="T_2cf59_row1_col1" class="data row1 col1" >26.27</td>
#           <td id="T_2cf59_row1_col2" class="data row1 col2" >35</td>
#           <td id="T_2cf59_row1_col3" class="data row1 col3" >6.57</td>
#           <td id="T_2cf59_row1_col4" class="data row1 col4" >76</td>
#           <td id="T_2cf59_row1_col5" class="data row1 col5" >14.26</td>
#           <td id="T_2cf59_row1_col6" class="data row1 col6" >46</td>
#           <td id="T_2cf59_row1_col7" class="data row1 col7" >8.63</td>
#         </tr>
#         <tr>
#           <th id="T_2cf59_level0_row2" class="row_heading
#           level0 row2" >True Positive</th>
#           <td id="T_2cf59_row2_col0" class="data row2 col0" >83</td>
#           <td id="T_2cf59_row2_col1" class="data row2 col1" >15.57</td>
#           <td id="T_2cf59_row2_col2" class="data row2 col2" >188</td>
#           <td id="T_2cf59_row2_col3" class="data row2 col3" >35.27</td>
#           <td id="T_2cf59_row2_col4" class="data row2 col4" >147</td>
#           <td id="T_2cf59_row2_col5" class="data row2 col5" >27.58</td>
#           <td id="T_2cf59_row2_col6" class="data row2 col6" >177</td>
#           <td id="T_2cf59_row2_col7" class="data row2 col7" >33.21</td>
#         </tr>
#         <tr>
#           <th id="T_2cf59_level0_row3" class="row_heading
#           level0 row3" >False Positive</th>
#           <td id="T_2cf59_row3_col0" class="data row3 col0" >32</td>
#           <td id="T_2cf59_row3_col1" class="data row3 col1" >6.00</td>
#           <td id="T_2cf59_row3_col2" class="data row3 col2" >31</td>
#           <td id="T_2cf59_row3_col3" class="data row3 col3" >5.82</td>
#           <td id="T_2cf59_row3_col4" class="data row3 col4" >61</td>
#           <td id="T_2cf59_row3_col5" class="data row3 col5" >11.44</td>
#           <td id="T_2cf59_row3_col6" class="data row3 col6" >43</td>
#           <td id="T_2cf59_row3_col7" class="data row3 col7" >8.07</td>
#         </tr>
#       </tbody>
#     </table>
#
#     </div>
#     <br />
#     <br />
##########################################################
#############################################
# Comparison of models:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We will compare now the histograms of the conditional probability for the
# two posible outcomes. Overlapping in the histograms means that the
# classification is erroneous. Condisering a cutoff of 0.5 pink cases below
# this point are false negatives (FN) and blue cases above the threhsold false
# positives (FP). It is important to consider that in this classification
# problem the missclassification of a patient with fatal outcome (FN) is
# considered a much more serious error.


def scatterPlot(df, ax):
    """
    Takes DF created with defDataFrame and creates a boxplot of
    different classification by mortal probability.
    """
    sns.swarmplot(
        ax=ax,
        y="Category",
        x="Probabilities",
        data=df,
        size=4,
        palette=sns.color_palette("tab10"),
        linewidth=0,
        dodge=False,
        alpha=0.6,
        order=[
            "True Negative",
            "False Negative",
            "True Positive",
            "False Positive",
        ],
    )
    sns.boxplot(
        ax=ax,
        x="Probabilities",
        y="Category",
        color="White",
        data=df,
        order=[
            "True Negative",
            "False Negative",
            "True Positive",
            "False Positive",
        ],
        saturation=15,
    )
    ax.set_xlabel("Probability of mortality")
    ax.set_ylabel("")


def plotHisto(df, ax, threshold=0.5, normalize=True):
    """
    Takes DF created with defDataFrame and plots histograms based on the
    probability of mortality by real Status at a selected @threshold.
    """
    if normalize:
        norm_params = {"stat": "density", "common_norm": False}
    else:
        norm_params = {}
    sns.histplot(
        ax=ax,
        data=df[df["Real"] == 1],
        x="Probabilities",
        color="deeppink",
        label="Deceased",
        bins=15,
        binrange=[0, 1],
        alpha=0.6,
        element="step",
        **norm_params
    )
    sns.histplot(
        ax=ax,
        data=df[df["Real"] == 0],
        x="Probabilities",
        color="dodgerblue",
        label="Survived",
        bins=15,
        binrange=[0, 1],
        alpha=0.4,
        element="step",
        **norm_params
    )
    ax.axvline(
        threshold, 0, 1, linestyle=(0, (1, 10)), linewidth=0.7, color="black"
    )


# visualize results
fig, ax = plt.subplots(
    nrows=2,
    ncols=2,
    sharex="all",
    sharey="all",
    gridspec_kw={"wspace": 0.1, "hspace": 0.35},
)
plotHisto(df_CMRC, ax=ax[0, 0], normalize=False)
ax[0, 0].set_title("CMRC")
plotHisto(df_MRC, ax=ax[1, 0], normalize=False)
ax[1, 0].set_title("MRC")
plotHisto(df_LR, ax=ax[0, 1], normalize=False)
ax[0, 1].set_title("LR")
ax[0, 1].legend()
plotHisto(df_SVC, ax=ax[1, 1], normalize=False)
ax[1, 1].set_title("SVC")
fig.tight_layout()

##############################################
# .. image:: images/images_COVID/COVID_001.png
#   :width: 600
#   :align: center
#   :alt: Imagen de prueba

##############################################
#############################################
# We see a clear different behaviour with the CMRC and MRC. MRC tends to
# estimate conditional probabilities in a more conservative way, rangin from
# 0.25 to 0.75. This estimation is very sensible to cut-off changes. The CMRC
# model shows a distribution where most of the cases are grouped around 0 and 1
# for survive and decease respectively. This results are similar to the
# Logistic Regression's but with less overlapping. SVC is the model with the
# worst performance of all having a lot of patients that survived with high
# decease probabilities.


cm_cmrc = confusion_matrix(y_test, clf_CMRC.predict(X_test))  # CMRC
cm_mrc = confusion_matrix(y_test, clf_MRC.predict(X_test))  # MRC
cm_lr = confusion_matrix(y_test, clf_LR.predict(X_test))  # Logistic Regression
cm_svc = confusion_matrix(
    y_test, clf_SVC.predict(X_test)
)  # C-Support Vector Machine

fig, ax = plt.subplots(
    nrows=2,
    ncols=2,
    sharex="all",
    sharey="all",
    gridspec_kw={"wspace": 0, "hspace": 0.35},
)
ConfusionMatrixDisplay(cm_cmrc, display_labels=["Survive", "Decease"]).plot(
    colorbar=False, ax=ax[0, 0]
)
ax[0, 0].set_title("CMRC")
ConfusionMatrixDisplay(cm_mrc, display_labels=["Survive", "Decease"]).plot(
    colorbar=False, ax=ax[1, 0]
)
ax[1, 0].set_title("MRC")
ConfusionMatrixDisplay(cm_lr, display_labels=["Survive", "Decease"]).plot(
    colorbar=False, ax=ax[0, 1]
)
ax[0, 1].set_title("LR")
ConfusionMatrixDisplay(cm_svc, display_labels=["Survive", "Decease"]).plot(
    colorbar=False, ax=ax[1, 1]
)
ax[1, 1].set_title("SVC")
fig.tight_layout()

##############################################
# .. image:: images/images_COVID/COVID_002.png
#   :width: 600
#   :align: center
#   :alt: Confusion Matrices

##############################################
#############################################
pd.DataFrame(
    classification_report(
        y_test,
        clf_CMRC.predict(X_test),
        target_names=["Survive", "Decease"],
        output_dict=True,
    )
).style.set_caption("Classification report CMRC").format(precision=3)

############################################
# .. raw:: html
#
#     <div class="output_subarea output_html rendered_html output_result">
#     <style type="text/css">
#     </style>
#     <table id="T_c03b8_">
#       <caption>Classification report CMRC</caption>
#       <thead>
#         <tr>
#           <th class="blank level0" >&nbsp;</th>
#           <th class="col_heading level0 col0" >Survive</th>
#           <th class="col_heading level0 col1" >Decease</th>
#           <th class="col_heading level0 col2" >accuracy</th>
#           <th class="col_heading level0 col3" >macro avg</th>
#           <th class="col_heading level0 col4" >weighted avg</th>
#         </tr>
#       </thead>
#       <tbody>
#         <tr>
#           <th id="T_c03b8_level0_row0" class="row_heading
# level0 row0" >precision</th>
#           <td id="T_c03b8_row0_col0" class="data row0 col0" >0.889</td>
#           <td id="T_c03b8_row0_col1" class="data row0 col1" >0.858</td>
#           <td id="T_c03b8_row0_col2" class="data row0 col2" >0.876</td>
#           <td id="T_c03b8_row0_col3" class="data row0 col3" >0.873</td>
#           <td id="T_c03b8_row0_col4" class="data row0 col4" >0.876</td>
#         </tr>
#         <tr>
#           <th id="T_c03b8_level0_row1" class="row_heading
# level0 row1" >recall</th>
#           <td id="T_c03b8_row1_col0" class="data row1 col0" >0.900</td>
#           <td id="T_c03b8_row1_col1" class="data row1 col1" >0.843</td>
#           <td id="T_c03b8_row1_col2" class="data row1 col2" >0.876</td>
#           <td id="T_c03b8_row1_col3" class="data row1 col3" >0.872</td>
#           <td id="T_c03b8_row1_col4" class="data row1 col4" >0.876</td>
#         </tr>
#         <tr>
#           <th id="T_c03b8_level0_row2" class="row_heading
# level0 row2" >f1-score</th>
#           <td id="T_c03b8_row2_col0" class="data row2 col0" >0.894</td>
#           <td id="T_c03b8_row2_col1" class="data row2 col1" >0.851</td>
#           <td id="T_c03b8_row2_col2" class="data row2 col2" >0.876</td>
#           <td id="T_c03b8_row2_col3" class="data row2 col3" >0.872</td>
#           <td id="T_c03b8_row2_col4" class="data row2 col4" >0.876</td>
#         </tr>
#         <tr>
#           <th id="T_c03b8_level0_row3" class="row_heading
# level0 row3" >support</th>
#           <td id="T_c03b8_row3_col0" class="data row3 col0" >310.000</td>
#           <td id="T_c03b8_row3_col1" class="data row3 col1" >223.000</td>
#           <td id="T_c03b8_row3_col2" class="data row3 col2" >0.876</td>
#           <td id="T_c03b8_row3_col3" class="data row3 col3" >533.000</td>
#           <td id="T_c03b8_row3_col4" class="data row3 col4" >533.000</td>
#         </tr>
#       </tbody>
#     </table>
#
#     </div>
#     <br />
#     <br />
#############################################
pd.DataFrame(
    classification_report(
        y_test,
        clf_MRC.predict(X_test),
        target_names=["Survive", "Decease"],
        output_dict=True,
    )
).style.set_caption("Classification report MRC").format(precision=3)
#############################################
# .. raw:: html
#
#     <div class="output_subarea output_html rendered_html output_result">
#     <style type="text/css">
#     </style>
#     <table id="T_31da6_">
#       <caption>Classification report MRC</caption>
#       <thead>
#         <tr>
#           <th class="blank level0" >&nbsp;</th>
#           <th class="col_heading level0 col0" >Survive</th>
#           <th class="col_heading level0 col1" >Decease</th>
#           <th class="col_heading level0 col2" >accuracy</th>
#           <th class="col_heading level0 col3" >macro avg</th>
#           <th class="col_heading level0 col4" >weighted avg</th>
#         </tr>
#       </thead>
#       <tbody>
#         <tr>
#           <th id="T_31da6_level0_row0" class="row_heading
# level0 row0" >precision</th>
#           <td id="T_31da6_row0_col0" class="data row0 col0" >0.665</td>
#           <td id="T_31da6_row0_col1" class="data row0 col1" >0.722</td>
#           <td id="T_31da6_row0_col2" class="data row0 col2" >0.677</td>
#           <td id="T_31da6_row0_col3" class="data row0 col3" >0.693</td>
#           <td id="T_31da6_row0_col4" class="data row0 col4" >0.689</td>
#         </tr>
#         <tr>
#           <th id="T_31da6_level0_row1" class="row_heading
# level0 row1" >recall</th>
#           <td id="T_31da6_row1_col0" class="data row1 col0" >0.897</td>
#           <td id="T_31da6_row1_col1" class="data row1 col1" >0.372</td>
#           <td id="T_31da6_row1_col2" class="data row1 col2" >0.677</td>
#           <td id="T_31da6_row1_col3" class="data row1 col3" >0.634</td>
#           <td id="T_31da6_row1_col4" class="data row1 col4" >0.677</td>
#         </tr>
#         <tr>
#           <th id="T_31da6_level0_row2" class="row_heading
# level0 row2" >f1-score</th>
#           <td id="T_31da6_row2_col0" class="data row2 col0" >0.764</td>
#           <td id="T_31da6_row2_col1" class="data row2 col1" >0.491</td>
#           <td id="T_31da6_row2_col2" class="data row2 col2" >0.677</td>
#           <td id="T_31da6_row2_col3" class="data row2 col3" >0.627</td>
#           <td id="T_31da6_row2_col4" class="data row2 col4" >0.650</td>
#         </tr>
#         <tr>
#           <th id="T_31da6_level0_row3" class="row_heading
# level0 row3" >support</th>
#           <td id="T_31da6_row3_col0" class="data row3 col0" >310.000</td>
#           <td id="T_31da6_row3_col1" class="data row3 col1" >223.000</td>
#           <td id="T_31da6_row3_col2" class="data row3 col2" >0.677</td>
#           <td id="T_31da6_row3_col3" class="data row3 col3" >533.000</td>
#           <td id="T_31da6_row3_col4" class="data row3 col4" >533.000</td>
#         </tr>
#       </tbody>
#     </table>
#
#     </div>
#     <br />
#     <br />
#############################################
pd.DataFrame(
    classification_report(
        y_test,
        clf_LR.predict(X_test),
        target_names=["Survive", "Decease"],
        output_dict=True,
    )
).style.set_caption("Classification report LR").format(precision=3)
#############################################
# .. raw:: html
#
#     <div class="output_subarea output_html rendered_html output_result">
#     <style type="text/css">
#     </style>
#     <table id="T_ecacf_">
#       <caption>Classification report LR</caption>
#       <thead>
#         <tr>
#           <th class="blank level0" >&nbsp;</th>
#           <th class="col_heading level0 col0" >Survive</th>
#           <th class="col_heading level0 col1" >Decease</th>
#           <th class="col_heading level0 col2" >accuracy</th>
#           <th class="col_heading level0 col3" >macro avg</th>
#           <th class="col_heading level0 col4" >weighted avg</th>
#         </tr>
#       </thead>
#       <tbody>
#         <tr>
#           <th id="T_ecacf_level0_row0" class="row_heading
# level0 row0" >precision</th>
#           <td id="T_ecacf_row0_col0" class="data row0 col0" >0.853</td>
#           <td id="T_ecacf_row0_col1" class="data row0 col1" >0.805</td>
#           <td id="T_ecacf_row0_col2" class="data row0 col2" >0.833</td>
#           <td id="T_ecacf_row0_col3" class="data row0 col3" >0.829</td>
#           <td id="T_ecacf_row0_col4" class="data row0 col4" >0.833</td>
#         </tr>
#         <tr>
#           <th id="T_ecacf_level0_row1" class="row_heading
# level0 row1" >recall</th>
#           <td id="T_ecacf_row1_col0" class="data row1 col0" >0.861</td>
#           <td id="T_ecacf_row1_col1" class="data row1 col1" >0.794</td>
#           <td id="T_ecacf_row1_col2" class="data row1 col2" >0.833</td>
#           <td id="T_ecacf_row1_col3" class="data row1 col3" >0.828</td>
#           <td id="T_ecacf_row1_col4" class="data row1 col4" >0.833</td>
#         </tr>
#         <tr>
#           <th id="T_ecacf_level0_row2" class="row_heading
# level0 row2" >f1-score</th>
#           <td id="T_ecacf_row2_col0" class="data row2 col0" >0.857</td>
#           <td id="T_ecacf_row2_col1" class="data row2 col1" >0.799</td>
#           <td id="T_ecacf_row2_col2" class="data row2 col2" >0.833</td>
#           <td id="T_ecacf_row2_col3" class="data row2 col3" >0.828</td>
#           <td id="T_ecacf_row2_col4" class="data row2 col4" >0.833</td>
#         </tr>
#         <tr>
#           <th id="T_ecacf_level0_row3" class="row_heading
# level0 row3" >support</th>
#           <td id="T_ecacf_row3_col0" class="data row3 col0" >310.000</td>
#           <td id="T_ecacf_row3_col1" class="data row3 col1" >223.000</td>
#           <td id="T_ecacf_row3_col2" class="data row3 col2" >0.833</td>
#           <td id="T_ecacf_row3_col3" class="data row3 col3" >533.000</td>
#           <td id="T_ecacf_row3_col4" class="data row3 col4" >533.000</td>
#         </tr>
#       </tbody>
#     </table>
#
#     </div>
#     <br />
#     <br />
#############################################
pd.DataFrame(
    classification_report(
        y_test,
        clf_SVC.predict(X_test),
        target_names=["Survive", "Decease"],
        output_dict=True,
    )
).style.set_caption("Classification report SVC").format(precision=3)
#############################################
# .. raw:: html
#
#     <div class="output_subarea output_html rendered_html output_result">
#     <style type="text/css">
#     </style>
#     <table id="T_0ab9f_">
#       <caption>Classification report SVC</caption>
#       <thead>
#         <tr>
#           <th class="blank level0" >&nbsp;</th>
#           <th class="col_heading level0 col0" >Survive</th>
#           <th class="col_heading level0 col1" >Decease</th>
#           <th class="col_heading level0 col2" >accuracy</th>
#           <th class="col_heading level0 col3" >macro avg</th>
#           <th class="col_heading level0 col4" >weighted avg</th>
#         </tr>
#       </thead>
#       <tbody>
#         <tr>
#           <th id="T_0ab9f_level0_row0" class="row_heading
# level0 row0" >precision</th>
#           <td id="T_0ab9f_row0_col0" class="data row0 col0" >0.743</td>
#           <td id="T_0ab9f_row0_col1" class="data row0 col1" >0.717</td>
#           <td id="T_0ab9f_row0_col2" class="data row0 col2" >0.734</td>
#           <td id="T_0ab9f_row0_col3" class="data row0 col3" >0.730</td>
#           <td id="T_0ab9f_row0_col4" class="data row0 col4" >0.732</td>
#         </tr>
#         <tr>
#           <th id="T_0ab9f_level0_row1" class="row_heading
# level0 row1" >recall</th>
#           <td id="T_0ab9f_row1_col0" class="data row1 col0" >0.829</td>
#           <td id="T_0ab9f_row1_col1" class="data row1 col1" >0.601</td>
#           <td id="T_0ab9f_row1_col2" class="data row1 col2" >0.734</td>
#           <td id="T_0ab9f_row1_col3" class="data row1 col3" >0.715</td>
#           <td id="T_0ab9f_row1_col4" class="data row1 col4" >0.734</td>
#         </tr>
#         <tr>
#           <th id="T_0ab9f_level0_row2" class="row_heading
# level0 row2" >f1-score</th>
#           <td id="T_0ab9f_row2_col0" class="data row2 col0" >0.784</td>
#           <td id="T_0ab9f_row2_col1" class="data row2 col1" >0.654</td>
#           <td id="T_0ab9f_row2_col2" class="data row2 col2" >0.734</td>
#           <td id="T_0ab9f_row2_col3" class="data row2 col3" >0.719</td>
#           <td id="T_0ab9f_row2_col4" class="data row2 col4" >0.729</td>
#         </tr>
#         <tr>
#           <th id="T_0ab9f_level0_row3" class="row_heading
# level0 row3" >support</th>
#           <td id="T_0ab9f_row3_col0" class="data row3 col0" >310.000</td>
#           <td id="T_0ab9f_row3_col1" class="data row3 col1" >223.000</td>
#           <td id="T_0ab9f_row3_col2" class="data row3 col2" >0.734</td>
#           <td id="T_0ab9f_row3_col3" class="data row3 col3" >533.000</td>
#           <td id="T_0ab9f_row3_col4" class="data row3 col4" >533.000</td>
#         </tr>
#       </tbody>
#     </table>
#
#     </div>
#     <br />
#     <br />
#############################################
############################################
# We can see in the classification reports and the confusion matrices the
# outperformance of CMRC.


#############################################
# Setting the cut-off point for binary classification:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this section we will use beeswarm-boxplot to select the cut-off point
# to optimise the tradeoff between false positives and false negatives. The
# beeswarm-boxplot is a great tool to determine the performance of the model
# in each of the cases of the confusion matrix. On an ideal scenario the errors
# are located near the cut-off point and the true guesses are located near the
# 0 and 1 values.
fig, ax = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(10, 12),
    sharex="all",
    sharey="all",
    gridspec_kw={"wspace": 0.1, "hspace": 0.20},
)
scatterPlot(df_CMRC, ax[0, 0])
ax[0, 0].set_title("CMRC")
scatterPlot(df_MRC, ax[1, 0])
ax[1, 0].set_title("MRC")
scatterPlot(df_LR, ax[0, 1])
ax[0, 1].set_title("LR")
scatterPlot(df_SVC, ax[1, 1])
ax[1, 1].set_title("SVC")
plt.tight_layout()

##############################################
# .. image:: images/images_COVID/COVID_003.png
#   :width: 600
#   :align: center
#   :alt: Imagen de prueba

##############################################

#############################################
# We see in the CMRC that the correct cases have a very good
# conditional probability estimation with around 75% of the cases very close to
# the extreme values. The most problematic cases are those with a low mortality
# probability estimation that had a fatal outcome (FN). In the CMRC
# model adjusting the threshold to 0.35 reduces the false negatives by 25%
# adding just some cases to the FP. In the MRC model adjusting the cutoff to
# 0.4 reduces half of the false negatives by trading of 25% of the TP.

threshold = 0.35
df_CMRC = defDataFrame(
    model=clf_CMRC, x_test=X_test, y_test=y_test, threshold=threshold
)
threshold = 0.4
df_MRC = defDataFrame(
    model=clf_MRC, x_test=X_test, y_test=y_test, threshold=threshold
)
pd.DataFrame(
    classification_report(
        df_CMRC.Real,
        df_CMRC.Prediction,
        target_names=["Survive", "Decease"],
        output_dict=True,
    )
).style.set_caption("Classification report CMRC \n adjusted threshold").format(
    precision=3
)

#############################################
# .. raw:: html
#
#     <div class="output_subarea output_html rendered_html output_result">
#     <style type="text/css">
#     </style>
#     <table id="T_a0276_">
#       <caption>Classification report CMRC
#      adjusted threshold</caption>
#       <thead>
#         <tr>
#           <th class="blank level0" >&nbsp;</th>
#           <th class="col_heading level0 col0" >Survive</th>
#           <th class="col_heading level0 col1" >Decease</th>
#           <th class="col_heading level0 col2" >accuracy</th>
#           <th class="col_heading level0 col3" >macro avg</th>
#           <th class="col_heading level0 col4" >weighted avg</th>
#         </tr>
#       </thead>
#       <tbody>
#         <tr>
#           <th id="T_a0276_level0_row0" class="row_heading
# level0 row0" >precision</th>
#           <td id="T_a0276_row0_col0" class="data row0 col0" >0.919</td>
#           <td id="T_a0276_row0_col1" class="data row0 col1" >0.800</td>
#           <td id="T_a0276_row0_col2" class="data row0 col2" >0.863</td>
#           <td id="T_a0276_row0_col3" class="data row0 col3" >0.859</td>
#           <td id="T_a0276_row0_col4" class="data row0 col4" >0.869</td>
#         </tr>
#         <tr>
#           <th id="T_a0276_level0_row1" class="row_heading
# level0 row1" >recall</th>
#           <td id="T_a0276_row1_col0" class="data row1 col0" >0.839</td>
#           <td id="T_a0276_row1_col1" class="data row1 col1" >0.897</td>
#           <td id="T_a0276_row1_col2" class="data row1 col2" >0.863</td>
#           <td id="T_a0276_row1_col3" class="data row1 col3" >0.868</td>
#           <td id="T_a0276_row1_col4" class="data row1 col4" >0.863</td>
#         </tr>
#         <tr>
#           <th id="T_a0276_level0_row2" class="row_heading
# level0 row2" >f1-score</th>
#           <td id="T_a0276_row2_col0" class="data row2 col0" >0.877</td>
#           <td id="T_a0276_row2_col1" class="data row2 col1" >0.846</td>
#           <td id="T_a0276_row2_col2" class="data row2 col2" >0.863</td>
#           <td id="T_a0276_row2_col3" class="data row2 col3" >0.861</td>
#           <td id="T_a0276_row2_col4" class="data row2 col4" >0.864</td>
#         </tr>
#         <tr>
#           <th id="T_a0276_level0_row3" class="row_heading
# level0 row3" >support</th>
#           <td id="T_a0276_row3_col0" class="data row3 col0" >310.000</td>
#           <td id="T_a0276_row3_col1" class="data row3 col1" >223.000</td>
#           <td id="T_a0276_row3_col2" class="data row3 col2" >0.863</td>
#           <td id="T_a0276_row3_col3" class="data row3 col3" >533.000</td>
#           <td id="T_a0276_row3_col4" class="data row3 col4" >533.000</td>
#         </tr>
#       </tbody>
#     </table>
#
#     </div>
#     <br />
#     <br />
#############################################
pd.DataFrame(
    classification_report(
        df_MRC.Real,
        df_MRC.Prediction,
        target_names=["Survive", "Decease"],
        output_dict=True,
    )
).style.set_caption("Classification report MRC \n adjusted threshold").format(
    precision=3
)
#############################################
# .. raw:: html
#
#     <div class="output_subarea output_html rendered_html output_result">
#     <style type="text/css">
#     </style>
#     <table id="T_bb0a1_">
#       <caption>Classification report MRC
#      adjusted threshold</caption>
#       <thead>
#         <tr>
#           <th class="blank level0" >&nbsp;</th>
#           <th class="col_heading level0 col0" >Survive</th>
#           <th class="col_heading level0 col1" >Decease</th>
#           <th class="col_heading level0 col2" >accuracy</th>
#           <th class="col_heading level0 col3" >macro avg</th>
#           <th class="col_heading level0 col4" >weighted avg</th>
#         </tr>
#       </thead>
#       <tbody>
#         <tr>
#           <th id="T_bb0a1_level0_row0" class="row_heading
# level0 row0" >precision</th>
#           <td id="T_bb0a1_row0_col0" class="data row0 col0" >0.811</td>
#           <td id="T_bb0a1_row0_col1" class="data row0 col1" >0.627</td>
#           <td id="T_bb0a1_row0_col2" class="data row0 col2" >0.715</td>
#           <td id="T_bb0a1_row0_col3" class="data row0 col3" >0.719</td>
#           <td id="T_bb0a1_row0_col4" class="data row0 col4" >0.734</td>
#         </tr>
#         <tr>
#           <th id="T_bb0a1_level0_row1" class="row_heading
# level0 row1" >recall</th>
#           <td id="T_bb0a1_row1_col0" class="data row1 col0" >0.665</td>
#           <td id="T_bb0a1_row1_col1" class="data row1 col1" >0.785</td>
#           <td id="T_bb0a1_row1_col2" class="data row1 col2" >0.715</td>
#           <td id="T_bb0a1_row1_col3" class="data row1 col3" >0.725</td>
#           <td id="T_bb0a1_row1_col4" class="data row1 col4" >0.715</td>
#         </tr>
#         <tr>
#           <th id="T_bb0a1_level0_row2" class="row_heading
# level0 row2" >f1-score</th>
#           <td id="T_bb0a1_row2_col0" class="data row2 col0" >0.730</td>
#           <td id="T_bb0a1_row2_col1" class="data row2 col1" >0.697</td>
#           <td id="T_bb0a1_row2_col2" class="data row2 col2" >0.715</td>
#           <td id="T_bb0a1_row2_col3" class="data row2 col3" >0.714</td>
#           <td id="T_bb0a1_row2_col4" class="data row2 col4" >0.717</td>
#         </tr>
#         <tr>
#           <th id="T_bb0a1_level0_row3" class="row_heading
# level0 row3" >support</th>
#           <td id="T_bb0a1_row3_col0" class="data row3 col0" >310.000</td>
#           <td id="T_bb0a1_row3_col1" class="data row3 col1" >223.000</td>
#           <td id="T_bb0a1_row3_col2" class="data row3 col2" >0.715</td>
#           <td id="T_bb0a1_row3_col3" class="data row3 col3" >533.000</td>
#           <td id="T_bb0a1_row3_col4" class="data row3 col4" >533.000</td>
#         </tr>
#       </tbody>
#     </table>
#
#     </div>
#     <br />
#     <br />
#############################################
# Results:
# -----------------
# Comparing the outputs of this example we can determine that MRCs work
# significantly well for estimating the outcome of COVID-19 patients at
# hospital triage.
#
# Furthermore, the CMRC model with threhsold feature mapping has shown a great
# performance both for classifying and for estimating conditional probabilities
# Finally we have seen how to select the cut-off values based on data
# visualization with beeswarm-boxplots to increase the recall in the desired
# class.
