# 🚗 Car Insurance Claim Prediction

This repository contains a full machine learning pipeline for predicting car insurance claims, including data processing, feature engineering, model tuning, and interpretability.

---

## 📊 Visualizations

### 1. Claim Rate by Age Group
![Claim Rate by Age](images/claim_by_age.png)  
📌 **Insight**: Younger and older age groups (especially 16–25 ) show higher claim rates.

### 2. Claim Rate by Driving Experience
![Claim Rate by Experience](images/claim_by_experience.png)  
📌 **Insight**: Drivers with 0–9 years of experience are more likely to file claims. Experience strongly correlates with lower risk.

### 3. PCA 3D Plot
![3D PCA Plot](images/pca_3d.png)  
📌 **Insight**: PCA reveals separability in clusters. The model likely finds distinct patterns between those who file claims and those who don’t.

### 4. SHAP Summary Plot (Real Results)
![SHAP Summary Plot](images/shap_summary.png)  
📌 **Insight**: Features like `DRIVING_EXPERIENCE_0-9y`, `VEHICLE_YEAR_before 2015`, and `CREDIT_SCORE` strongly influence predictions. SHAP values confirm feature importance rankings.

### 5. Neural Network Training Accuracy (Real Results)
![Training History](images/nn_accuracy_plot.png)  
📌 **Insight**: Model trains well and generalizes reasonably with early stopping. Minor fluctuations are due to dropout and validation split variation.

---

## 📂 Project Structure

- `Car_Insurance_Claim.csv` – The dataset
- `Car_Insurance_Colab_With_VisualExport.ipynb` – Full notebook with export-ready plots
- `images/` – All high-quality visualizations

---

## 🧠 Models Used

- **Random Forest**
- **PCA + Clustering**
- **Neural Network (Keras)**
- **Feature Selection (embedded)**
- **SHAP Explainability**

---

## 🧪 Evaluation

- Confusion Matrix
- Classification Report
- SHAP Summary
- Accuracy Curve

---

## 📌 Author

**Salah Aburajab**  
_Machine Learning & Data Science_

---

> A portfolio-ready ML project designed to showcase technical skill, end-to-end design, and interpretability best practices.
