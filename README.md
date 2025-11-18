# loan-default-shap-explainability
SHAP explainability for loan default prediction using XGBoost. Includes global and local SHAP analysis, stability testing, narrative case studies, and a complete interpretability report.
README – SHAP Explainability Project
Project Overview
This project applies SHAP (Shapley Additive exPlanations) to interpret an XGBoost machine learning model built for predicting loan default risk. The goal is to provide transparent global and local explanations, evaluate stability, and generate actionable insights.
Project Structure
shap_project/
├── data/ (Dataset)
├── models/ (Trained model)
├── outputs/ (Figures, plots, reports)
├── src/ (Python scripts)
├── notebooks/ (Optional Jupyter notebook)
├── SHAP_Project_Report.docx
└── README.docx
Model Details
• Model: XGBoost Classifier
• Accuracy: 93.69%
• Recall (Default class=1): 0.75
• F1-score (Default class=1): 0.84
Explainability Outputs
Global SHAP:
 - summary_plot.png
 - feature_importance.csv

Local SHAP:
 - waterfall plots
 - force plots (HTML)

Stability Analysis:
 - SHAP ranking stability
 - Spearman rho = 0.9966
Top 10 SHAP Features
1. person_income
2. loan_percent_income
3. loan_int_rate
4. person_home_ownership_RENT
5. loan_intent_VENTURE
6. loan_amnt
7. person_home_ownership_OWN
8. loan_grade_D
9. person_emp_length
10. person_age
How to Run
1. Train model:
   python src/train_model.py

2. Generate SHAP analysis:
   python src/shap_analysis.py

3. Run stability test:
   python src/stability_test.py

4. Generate narratives:
   python src/narratives.py
Actionable Insights
• High loan-to-income ratio is the strongest default predictor.
• Interest rate strongly reflects borrower risk.
• Home ownership plays a major role in repayment stability.
Limitations
• SHAP measures correlation, not causation.
• One-hot encoding splits importance across categories.
• Class imbalance may produce false negatives.
Credits
Developed by: MohammadAamir A

