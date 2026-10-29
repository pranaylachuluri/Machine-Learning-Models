# Machine Learning Models

Four supervised-learning projects from my Data Science internship at YHills (2023). Each notebook follows the same workflow: load the data, explore and clean it, visualise it, select features, train several scikit-learn models, and compare them on a held-out test set.

## Projects

| Notebook | Task | Best model | Test result |
| --- | --- | --- | --- |
| [Fraudulent Transactions](Fraudulent%20Transactions%20.ipynb) | Flag fraudulent payments (classification) | XGBoost | 99.2% accuracy |
| [Vaccine project](Vaccine%20project.ipynb) | Predict whether a person took the H1N1 vaccine (classification) | SVM / AdaBoost | 82.6% accuracy |
| [Taxi fare prediction](Taxi%20fare%20prediction%20model.ipynb) | Predict a taxi fare from pickup and drop-off details (regression) | Random Forest | R² 0.76, RMSE 2.58 |
| [Music Taste Prediction](Music%20Taste%20Prediction.ipynb) | Predict music genre from age and gender (classification) | Decision Tree | 75% accuracy |

### 1. Fraudulent transaction detection
Compares Logistic Regression (90.0%), Decision Tree (98.3%), Random Forest (98.7%) and XGBoost (99.2%).

### 2. H1N1 vaccination prediction
Uses survey answers about behaviour, opinions and demographics. Seven classifiers were compared: Logistic Regression, Decision Tree, Random Forest, Bagging, AdaBoost, Gradient Boosting and SVM. Most reach about 82% accuracy. Only about a fifth of people were vaccinated, so recall on that class (0.27 to 0.39) is the harder number to improve.

Dataset: [h1n1_vaccine_prediction.csv](https://github.com/pranaylachuluri/Machine-Learning-Models/files/12888182/h1n1_vaccine_prediction.csv)

### 3. Taxi fare prediction
Seven regressors were compared on R², RMSE, MAE and MSLE. The tree ensembles (Random Forest 0.76, AdaBoost 0.76, Gradient Boosting 0.75, Bagging 0.74) clearly beat Linear Regression and SVR (R² near 0), which shows the fare depends on location features non-linearly.

Dataset: [TaxiFare.csv](https://github.com/pranaylachuluri/Machine-Learning-Models/files/12888068/TaxiFare.csv)

### 4. Music taste prediction
A small Decision Tree example that predicts a listener's preferred genre from age and gender.

Dataset: [music.csv](https://github.com/pranaylachuluri/Machine-Learning-Models/files/12888218/music.csv)

## Tech

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Jupyter

## Run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
jupyter notebook
```

Download the dataset linked for each project and update the `read_csv` path at the top of its notebook.
