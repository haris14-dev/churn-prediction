# Telco Customer Churn Prediction

## Project Overview
Customer churn occurs when a customer stops using a company's services. For telecom companies, churn results in significant revenue loss.  
This project predicts which customers are likely to churn using **Logistic Regression**, enabling proactive retention strategies.

**Goal:**  
- Predict customer churn  
- Maximize recall for churners to save revenue  
- Balance false alarms to minimize unnecessary costs

---

## Dataset
- Source: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- Rows: ~7,000+ 
- Features: 20+ customer attributes (numerical, categorical, binary)  
- Target: `Churn` (Yes = 1, No = 0)

**Preprocessing Steps:**  
- Label encoding for binary columns (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`)  
- One-hot encoding for categorical features (`Contract`, `PaymentMethod`, `InternetService`, etc.)  
- Scaling numerical features using `StandardScaler`  
- Handling missing values in `TotalCharges`  

---

## Model Used
**Algorithm:** Logistic Regression  

**Hyperparameters:**
- `max_iter=1000` → ensures convergence  
- `class_weight='balanced'` → handles class imbalance  
- `C=100` → aggressive regularization  

**Threshold Tuning:**
- Optimized threshold = **0.55** → balances recall and precision

## Evaluation Metrics

- ### Confusion Matrix (Threshold = 0.55)
 [[777 258]
 [ 93 281]]

- **TP = 281** → churners correctly identified  
- **FN = 93** → churners missed  
- **FP = 258** → loyal customers incorrectly flagged  
- **TN = 777** → loyal customers correctly predicted  

### Classification Report
| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.89      | 0.75   | 0.82     | 1035    |
| 1     | 0.52      | 0.75   | 0.62     | 374     |

**Key Insight:**  
- Recall for churn = 0.75 → 75% of churners caught ✅  
- Precision = 0.52 → acceptable false alarms (trade-off for recall)  
- Accuracy = 75–85% → secondary metric due to class imbalance  

---

## Business Impact
- **High recall** ensures most churners are identified → revenue saved  
- **Moderate precision** → some loyal customers contacted → manageable cost  
- Threshold tuning ensures **practical trade-off** between recall and precision

---

## Visual Confusion Matrix

You can generate a **heatmap** using this code:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_custom)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Churn','Churn'], yticklabels=['Non-Churn','Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



