# 💳 Creditworthiness Predictor

A Streamlit-based web application that predicts the creditworthiness of individuals using the **German Credit Data** and a trained **Random Forest model**.

> ✅ Built during the **Celebal Summer Internship – Data Science Domain**

🔗 **Live App**:  
👉 [https://gangakatte-celebalsummerinternship-csi-finalprojectapp-02x7oa.streamlit.app](https://gangakatte-celebalsummerinternship-csi-finalprojectapp-02x7oa.streamlit.app)

---

## 🧠 What it does

This app uses your financial details to predict:
- ✅ If you are **Creditworthy**, or
- ❌ **Not Creditworthy**

It also shows possible **reasons for rejection** to help you understand and improve.

---

## 🔍 Features

- Intuitive, mobile-friendly UI built with **Streamlit**
- Real-time prediction using a trained **Random Forest** model
- Scales and preprocesses data automatically
- User-friendly dropdowns and tooltips for guidance
- Styled with modern fonts and colors
- Explains *why* a user is or isn’t eligible for credit

---

## 📊 Technologies Used

| Tool            | Role                       |
|-----------------|----------------------------|
| Python          | Core logic                 |
| Streamlit       | UI + Deployment            |
| Scikit-learn    | Machine Learning           |
| NumPy, Pandas   | Data preprocessing         |
| GitHub          | Version control + hosting  |

---

## 🏗 Project Structure

```
📁 CSI_FinalProject
├── app.py                         # Streamlit frontend
├── creditworthiness_rf_model.pkl # Trained ML model
├── requirements.txt              # Dependencies

```

---

## 🚀 Run Locally

1. Clone the repo:

```bash
git clone https://github.com/GangaKatte/CelebalSummerInternship.git
cd CelebalSummerInternship/CSI_FinalProject
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

---

## 📘 Dataset Note

- Based on the **Statlog (German Credit Data)** from UCI Repository  
- Currency used in the dataset is **DM (Deutsche Mark)**  
  ➤ 1 DM = 0.511 Euro ≈ ₹46 INR (as of 2025)

---

## 🙋‍♀️ Author

**Ganga Katte**  
B.Tech – Information Technology  
Walchand Institute of Technology  
🔗 [LinkedIn](https://www.linkedin.com/in/gangakatte/)

---

## ⭐ Feedback or Support

If you found this helpful or impressive, please ⭐ star the repo and share the app with others!

---
