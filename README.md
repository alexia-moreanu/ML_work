# ML_work

## Titanic Multivariate Linear Regression (CS156 PreClassWork)

[`Titanic_Multivariate_Regression.ipynb`](Titanic_Multivariate_Regression.ipynb)
works through the linear algebra ↔ multivariate linear regression PreClassWork
assignment: predicting Titanic passenger **fare** from `Pclass`, `Age`, and `SibSp`.

Contents:
- Conceptual Q&A on vector spaces, matrix operations, and least squares (Q1)
- A full `pandas`/`scikit-learn` pipeline: dataframe construction, train/test split,
  and a fitted `LinearRegression` model (Code Prep)
- Matrix-notation derivation of the OLS loss function, its derivative, and the
  normal-equations solution, plus a worked 3×3 numerical example (Q2)
- 3D and partial-dependence plots of the fitted regression plane (Q3/Q4)
- Sum of squared/absolute residuals computed by hand and in code, with the
  gradient of the loss function (Q5)

Data: `titanic/train.csv`, `titanic/test.csv`, `titanic/gender_submission.csv` from
the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic/data).
