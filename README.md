# Task  - Iris Flower Classification

## Intern Information

| Field             | Details                          |
|-------------------|----------------------------------|
| **Name**          | Khushi Kumari                    |
| **Intern ID**     | CITS2079                         |
| **Company**       | CodTech IT Solutions Pvt. Ltd.   |
| **Domain**        | Artificial Intelligence          |
| **Task**          | Iris Flower Classification       |
| **Mentor**        | Neela Santhosh Kumar             |
| **Duration**      | 4 Weeks                          |


---

## Objective

Build a machine learning model to classify Iris flowers into three species —
**Iris-setosa**, **Iris-versicolor**, and **Iris-virginica** — based on four
numerical features: sepal length, sepal width, petal length, and petal width.

---

## Dataset

- **File:** `iris.csv`
- **Source:** UCI Machine Learning Repository (Iris Dataset)
- **Rows:** 150 | **Columns:** 5
- **Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
- **Target:** `species` (3 classes, 50 samples each)

---

## Project Structure

```
Task-1-Iris-Classification/
│
├── iris.csv                  # Dataset file
├── iris_classification.py    # Main Python code
├── README.md                 # Project documentation
│
└── Output Plots (generated on running the code)
    ├── pairplot.png
    ├── correlation_heatmap.png
    ├── boxplots.png
    ├── confusion_matrix.png
    └── model_comparison.png
```

---

## Libraries Used

| Library       | Purpose                          |
|---------------|----------------------------------|
| pandas        | Data loading and manipulation    |
| numpy         | Numerical operations             |
| matplotlib    | Data visualization               |
| seaborn       | Statistical plots                |
| scikit-learn  | ML models and evaluation metrics |

---

## Models Trained

Four classification algorithms were trained and compared:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Support Vector Machine (SVM)**

---

## Steps Performed

1. Loaded and explored the dataset (shape, statistics, class distribution)
2. Checked for missing values
3. Visualized data using pairplots, boxplots, and correlation heatmap
4. Encoded target labels using `LabelEncoder`
5. Split data into 80% train and 20% test sets
6. Trained four ML models and compared accuracy
7. Generated classification report and confusion matrix for the best model
8. Tested the model on a new sample input

---

## Results

| Model                   | Accuracy  |
|-------------------------|-----------|
| Logistic Regression     | ~96.67%   |
| Decision Tree           | ~96.67%   |
| K-Nearest Neighbors     | ~96.67%   |
| Support Vector Machine  | ~96.67%   |

> Actual accuracy values are printed when you run the code.

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run the script
```bash
python iris_classification.py
```

The script will print results in the terminal and save all plot images in
the same folder.

---

## Sample Output

```
==================================================
       IRIS FLOWER CLASSIFICATION
==================================================
[1] Dataset Overview
    Shape : (150, 5)
...
[13] Sample Prediction on New Data
    Input  : [5.1, 3.5, 1.4, 0.2]
    Predicted Species : Iris-setosa
==================================================
  Task Completed Successfully!
==================================================
```

---

## Conclusion

This task demonstrates a complete end-to-end machine learning pipeline —
from data loading and EDA to model training, evaluation, and prediction —
using the classic Iris dataset. All four models achieved high accuracy,
confirming that the Iris dataset is linearly and non-linearly separable
with clean features.

---

*Submitted as part of CodTech IT Solutions Artificial Intelligence Internship — Task 1*
