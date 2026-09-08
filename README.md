# Detecting Fake Political Statements with PySpark MLlib

## Overview
It is easy in the modern day to spread false statements and for people to interpret the statements as true. Media sites, tv airings of speeches, interviews, emails and other ways of spreading news are a part of our daily lives. Politicians and figures have extreme power in manipulating what is considered true and what is considered fake news. A system of monitoring disinformation that can handle the massive amounts of digital data and digital methods of spreading statements from a politician is crucial to prevent harm to citizens and corruption by politicians. Therefore, the research question is: how can machine learning detect fake news in political statements that are accessed and stored digitally? Specifically, this project investigates data that originates from PolitiFact's online records. PolitiFact is a "fact-checking website that rates the accuracy of claims by elected officials and others" (PolitiFact).

The dataset utilized in this investigation is the "LIAR" dataset, created by William Wang in 2017. This dataset compiles data from over a decade of 12,800 manually labeled short statements in various contexts from PolitiFact.com. The original labels for categories of truthfulness are: true, mostly true, half true, barely true, false, and pants on fire (extremely false). The other variables included are the id of the statement JSON file, text of the statement, subject of the statement, speaker's name, speaker's job title, state, speaker's party affiliation, location or online place the statement was made, and the speaker's total credit history count (barely true, false, half true, mostly true pants on fire counts). The target variable used is a binary label where the original labels of true, mostly-true and half-true are considered true and the others as false. A binary target was of more interest rather than the 6 different categories as the applications could be more useful for broadly detecting fake statements.

In a big data context, this problem is relevant as it is easier than ever for misinformation to spread online with social media apps, news sites, videos and more. The amounts of data related to political statements are massive and no one human can go through each and every one to check for misinformation. The rate at which politicians make statements that are spread around and analyzed online is too quick to fully address without a machine. Additionally, human labeling could be biased if the person determining if a statement is true or false has a political preference. Detecting false news is harder than ever when a statement can be determined to be fake by one group of people and true by another. Thus, methods to handle and model big data are needed to solve this issue of detecting false political statements.

## Data
- **Source:** LIAR dataset (Wang, 2017) — 12,800+ manually labeled political statements from PolitiFact
- **Target:** Binary label (true / mostly-true / half-true → True; barely-true / false / pants-on-fire → False)
- **Features:** statement text, subject, speaker, speaker's job title, state, party affiliation, context and speaker credit history counts

## Methods
- **Preprocessing:** PySpark schema definition, missing value handling, categorical cleanup (state name standardization, etc.)
- **Text features:** Tokenizer → StopWordsRemover → HashingTF → TF-IDF
- **Categorical features:** StringIndexer + OneHotEncoder (low-cardinality: state, party); FeatureHasher (high-cardinality: subject, context, speaker, speaker_job)
- **Numerical features:** VectorAssembler + StandardScaler on speaker credit history counts
- **Models:**
  - Baseline: Decision Tree Classifier (PySpark MLlib)
  - Tuned Decision Tree (grid search + cross-validation)
  - Bagging ensemble of deep Decision Trees (30 bootstrapped learners, majority vote)

## Repo Structure
| File | Description |
|---|---|
| `DetectingFakeStatements_Code.ipynb` | Main analysis notebook (pipeline, modeling, evaluation) |
| `DetectingFakeStatements_Code.html` | HTML version of the notebook |
| `DetectingFakeStatements_Report.pdf` | Full report |

## Built With
`PySpark / Spark MLlib` · `Python` · `pandas`

## Key Takeaways
- Baseline Decision Tree: Accuracy 0.696, AUC-ROC 0.621, F1 0.680
- Hyperparameter-tuned Decision Tree: Accuracy 0.710, AUC-ROC 0.736, F1 0.705
- Bagging ensemble (30 deep trees): Accuracy 0.708, AUC-ROC 0.787 (best), F1 0.704
- Both advanced models meaningfully improved class separation (AUC-ROC) over the baseline. Bagging performed best overall at distinguishing true from false statements.
- A data-size scaling comparison showed the tuned decision tree consistently achieved the highest F1 score, while the bagging model scaled most efficiently in training time.

*For full methodology, EDA and evaluation details see `DetectingFakeStatements_Report.pdf`.*
