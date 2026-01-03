# 💻 Laptop Price Prediction — End-to-End Machine Learning Project  
### 🔥 Automated ML Pipeline | MySQL Ingestion | Model Training | Evaluation | Deployment | Streamlit UI

This is a complete **End-to-End Machine Learning project** that predicts laptop prices using historical data stored in **MySQL database**.  
The project contains everything from raw data ingestion → preprocessing → model training → evaluation → model versioning → prediction interface (Streamlit Web App).

---

# 🚀 Features

### ✅ **1. Automated ML Pipeline (`main.py`)**
- Data ingestion from MySQL  
- Data validation  
- Data transformation (scaling + encoding)  
- Model training (Linear Regression + Random Forest)  
- Model evaluation (RMSE, MAE, R²)  
- Model registry / deployment  
- Artifact versioning

### ✅ **2. Streamlit App (`app.py`)**
- Dropdown-based input form  
- Batch prediction (CSV upload)  
- Downloadable output  

### ✅ **3. Model Versioning**
Saved inside:
```
prediction/models/current_model.joblib
prediction/models/best_model_<timestamp>.joblib
```

### ✅ **4. Batch Testing**
Using:
```
python check_model.py
```

---

# 📁 Project Structure

```
📦 root
│
├── app.py
├── main.py
├── check_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── Dockerfile
├── .env (ignored)
│
├── laptop_price/
│   ├── components/
│   ├── pipeline/
│   ├── utils/
│   ├── entity/
│   ├── exception.py
│   ├── logger.py
│   └── config.py
│
├── artifacts/
│   ├── raw/
│   ├── transformed/
│   └── model/
│
└── prediction/
    └── models/
```

---

# 🛠️ Installation

## 1️⃣ Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/Laptop-Price-Prediction.git
cd Laptop-Price-Prediction
```

## 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

## 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

## 4️⃣ Create `.env`
```
DB_USER=root
DB_PASSWORD=
DB_HOST=localhost
DB_PORT=3306
DB_NAME=laptop_data
DB_TABLE=laptop_price
```

---

# ▶️ Run Training Pipeline
```bash
python main.py
```

---

# 🧪 Test the Model
```bash
python check_model.py
```

---

# 🎨 Streamlit App
```bash
streamlit run app.py
```

---

# 📊 Example Performance
```
Train R²: 0.965  
Test R²:  0.965  
RMSE:     8243.50  
MAE:      5749.01  
```

---

# 🌐 Deploy Options
- Streamlit Cloud
- Docker
- AWS / Render / Azure

---

# 🔒 Security Notes
- Do NOT push `.env`
- Avoid pushing large artifacts/models

---

# ⭐ Support
If you like this project, give it a ⭐ on GitHub!