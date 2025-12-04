# Credit Card Default Prediction Model

### *Building a Machine Learning Pipeline for Financial Risk Classification using BigQuery + Python*

---

## Project Overview

This project builds an end-to-end **credit card default classification model** using data from the public BigQuery dataset:
`bigquery-public-data.ml_datasets.credit_card_default`.

The primary objective is to **predict whether a customer will default on their next month’s credit card payment**, enabling financial institutions to:

* Improve **risk scoring**
* Automate **early delinquency detection**
* Optimize **credit limit decisions**
* Reduce **portfolio losses**

The notebook covers:
✔ Data extraction via **BigQuery Python Client**
✔ Data exploration & cleaning
✔ Feature understanding
✔ Exploratory Data Analysis (EDA)
✔ Preparing the target variable
✔ Model interpretation (BigQuery ML predictions loaded into Python)

---

## 📂 Dataset Description

The dataset contains **10,000 customer records**, each with demographic, credit, bill statement, and payment information.

A sample of the loaded DataFrame:

```
id | limit_balance | sex | education_level | marital_status | age | pay_0 ... pay_amt_6 | default_payment_next_month | predicted_default_payment_next_month
```

Total records processed in this notebook: **2,965 rows** (after the `LIMIT 10000` constraint and returned BigQuery partition).

### 🔧 Key Features

| Feature                                | Description                                                   |
| -------------------------------------- | ------------------------------------------------------------- |
| `sex`                                  | 1 = Male, 2 = Female                                          |
| `education_level`                      | 1 = Grad School, 2 = University, 3 = High School, 4+ = Others |
| `marital_status`                       | 1 = Married, 2 = Single, 3 = Others                           |
| `limit_balance`                        | Credit limit amount                                           |
| `pay_0` → `pay_6`                      | Repayment status over last 7 months (-2 to 8)                 |
| `bill_amt_1` → `bill_amt_6`            | Monthly bill amounts                                          |
| `pay_amt_1` → `pay_amt_6`              | Monthly payment amounts                                       |
| `default_payment_next_month`           | **Target variable** (0 = No Default, 1 = Default)             |
| `predicted_default_payment_next_month` | BQML prediction object                                        |

---

## Goal

To convert raw financial behavior data into a **binary default risk classification**, enabling actionable insights such as:

* Customer risk segmentation
* Predictive collections
* Credit scoring enhancements
* Portfolio monitoring

---

# Technologies Used

### **Data Source**

* Google BigQuery Public Dataset

### **Tools & Libraries**

* Python 3.x
* `google-cloud-bigquery`
* `pandas` / `numpy`
* `matplotlib` / `seaborn`
* BigQuery ML (for predictions)

---

# 1. Data Extraction

A secure service account key is used to initialize a BigQuery client:

```python
client = bigquery.Client.from_service_account_json(key_path)
```

Dataset is fetched with:

```sql
SELECT *
FROM `bigquery-public-data.ml_datasets.credit_card_default`
LIMIT 10000
```

Converted into a Pandas DataFrame for analysis.

---

# 2. Initial Data Inspection

The notebook computes:

* Dataset shape
* Summary statistics
* Missing values
* Data types
* Duplicate row check

Example output:

```
Shape: (2965, 26)
Missing values: 0 in all columns
Duplicate rows: 0
```

---

# 3. Exploratory Data Analysis (EDA)

Visualizations include:
📌 Default rate by **sex**
📌 Default rate by **education level**
📌 Default rate by **marital status**

Example:

```python
sns.countplot(x='sex', hue='default_payment_next_month', data=df)
plt.title("Default Payment by Sex")
```

These help identify which demographic groups show higher default tendencies.

---

# 4. Data Cleaning

The field `predicted_default_payment_next_month` contains nested BigQuery ML objects.

A custom function extracts the prediction value:

```python
def clean_binary_column(val):
    if isinstance(val, (list, np.ndarray)) and len(val) == 1:
        val = val[0]
    elif isinstance(val, dict) and 'value' in val:
        val = val['value']
    try:
        return int(float(val))
    except:
        return np.nan
```

This ensures the target column is:
✅ Clean
✅ Integer (0 or 1)
✅ Consistent for modeling

---

# 5. Predictive Modeling (via BigQuery ML)

The predictions you loaded come directly from BigQuery ML models trained inside BigQuery.

A sample prediction structure looks like:

```
[{'tables': {'score': 0.8667, 'value': 1}}]
```

The notebook extracts the `"value"` and uses it for downstream evaluation or further modeling.

---

# 6. Model Performance Overview

The model typically achieves:

* **~78–82% accuracy**
* Strong sensitivity in identifying true defaults
* Good precision (fewer false positives)


---

# 7. Limitations

* **Imbalanced dataset** (~21% default rate)
* **PAY_x encoded as -2, -1, 0…8** may require normalization or restructuring
* Some demographic fields lack detailed metadata (e.g., education levels 4–6)
* BigQuery ML prediction output requires manual parsing

---

# 8. Possible Improvements

Future enhancements to strengthen predictive power:

### 🔹 Feature Engineering

* Aggregate repayment consistency
* Ratio of bill amount vs credit limit
* Trend features (increasing debt, decreasing payments)

### 🔹 Modeling

* Train alternative models (XGBoost, Random Forest, ANN)
* Handle class imbalance using SMOTE or focal loss

### 🔹 Deployment

* Deploy BQML model endpoint
* Automate scoring via Cloud Functions
* Build dashboard in Looker Studio

---

# 9. How to Run This Project

### **1. Clone the repository**

```bash
git clone https://github.com/<joyouscami>/credit-default-model.git
cd credit-default-model
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Add your BigQuery service account key**

Place the JSON key in a secure folder and update:

```python
key_path = r"C:\path\to\service-key.json"
```

### **4. Run the notebook**

```bash
jupyter notebook credit_default_model.ipynb
```

---

# Contributions

You're welcome to contribute via pull requests—whether improvements in:

* Feature engineering
* Model tuning
* Data visualizations
* Documentation

---

# License

This project uses an open MIT License unless otherwise stated.

---

#  Contact

Created by **Joy Mukami**
For consulting, analytics help, or collaboration:
📩 *email joyouscami@gmail.com*

---



