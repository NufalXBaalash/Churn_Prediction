# 📊 Churn Prediction Using Customer Dataset

This project aims to **predict customer churn** — whether a customer is likely to stop using a service — using a labeled dataset. The goal is to build a machine learning model that can accurately classify customers based on their behavior and demographic profile.

---

## 📁 Dataset

Two CSV files are used:

* **Training Data**: `churn-bigml-80.csv` — 80% of the full dataset.
* **Testing Data**: `churn-bigml-20.csv` — remaining 20%, for evaluation.

---

## 🧱 Project Structure

```
churn_model.ipynb     # Main Jupyter Notebook with full code and outputs
churn-bigml-80.csv    # Training dataset
churn-bigml-20.csv    # Testing dataset
README.md             # Project overview and instructions
```

---

## 🧪 Features Overview

The dataset includes features such as:

* **State, Area Code, International Plan**
* **Voice Mail Plan, Number Vmail Messages**
* **Total Day/Evening/Night/International Minutes & Charges**
* **Customer Service Calls**
* **Target Variable**: `Churn` (Yes/No)

---

## 🧹 Preprocessing Steps

1. **Outlier Handling**:

   * IQR method is used to treat/remove outliers in numerical columns.

2. **Feature Scaling**:

   * `StandardScaler` is applied to normalize numerical features.

3. **Encoding**:

   * Categorical columns are encoded using `LabelEncoder`.

4. **Class Imbalance Handling**:

   * **SMOTE** (Synthetic Minority Over-sampling Technique) is used to balance the dataset.

---

## 🧠 Model Building

The project uses a **Neural Network (TensorFlow/Keras)** with the following architecture:

* `Dense(64)` with ReLU + `Dropout(0.3)`
* `Dense(32)` with ReLU
* `Dense(1)` with Sigmoid (for binary classification)

The model is compiled using:

* **Loss**: Binary Crossentropy
* **Optimizer**: Adam
* **Metrics**: Accuracy

---

## 📊 Evaluation

Model performance is evaluated using:

* Accuracy
* Loss curves (over epochs)
* Classification Report (Precision, Recall, F1-score)
* Confusion Matrix

---

## 📈 Visualizations

The notebook includes plots for:

* Training vs Validation Accuracy/Loss
* Feature importances (for tree-based models)
* Data distribution before/after preprocessing

---

## ▶️ How to Run

1. Clone the repository or download the notebook and dataset files.
2. Install required libraries:

   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn tensorflow matplotlib seaborn
   ```
3. Run `churn_model.ipynb` step-by-step.

---

## 💡 Future Improvements

* Hyperparameter tuning (GridSearch, RandomSearch)
* Ensemble methods (e.g., XGBoost, Random Forest)
* Model export for API/production use

---

## 🧑‍💻 Author

Built by **\[Ahmed Baalash]** — feel free to connect for collaboration or feedback!

