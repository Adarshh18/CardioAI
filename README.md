<div align="center">

<!-- HEADER BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=CardioAI&fontSize=80&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Smart%20Heart%20Risk%20Predictor&descAlignY=60&descSize=22" width="100%"/>

<br/>

<!-- BADGES -->
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-00f5ff?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-00d26a?style=for-the-badge)](https://github.com)

<br/>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Syne&weight=700&size=22&pause=1000&color=00F5FF&center=true&vCenter=true&width=600&lines=AI-Powered+Cardiovascular+Risk+Assessment;K-Nearest+Neighbors+Classification;918+Patient+Records+%7C+90%25+Accuracy;Built+with+Streamlit+%26+scikit-learn" alt="Typing SVG" />
</p>

</div>

---

## 🫀 What is CardioAI?

**CardioAI** is an intelligent, web-based cardiovascular risk prediction tool powered by machine learning. It analyzes **11 clinical parameters** — from age and blood pressure to ECG results and ST slope — and instantly predicts whether a patient is at **high or low risk** of heart disease.

Built with a sleek dark-themed UI using Streamlit, CardioAI combines the power of a **K-Nearest Neighbors classifier** trained on 918 real patient records with a modern, production-grade interface.

> ⚠️ **Disclaimer:** CardioAI is built for educational and research purposes only. It is not a certified medical device and must not replace professional diagnosis or treatment.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Instant Prediction** | Get heart risk assessment in milliseconds |
| 🎯 **90% Accuracy** | KNN model trained on 918 clinical records |
| 🔬 **11 Parameters** | Comprehensive cardiac feature analysis |
| 🛡️ **Privacy First** | All processing is done locally — no data stored |
| 💡 **Health Tips** | Evidence-based cardiovascular health guidance |
| 🌐 **Responsive UI** | Clean, modern dark-themed Streamlit interface |
| 📊 **Multi-dataset** | Combines 5 international heart disease datasets |

---

## 🖥️ App Preview

<div align="center">

### 🏠 Home Page
```
╔══════════════════════════════════════════════════════╗
║  🫀 CardioAI          Home  Prediction  Tips  About  ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║          ● AI-POWERED CARDIAC ANALYSIS               ║
║                                                      ║
║          Smart Heart Risk                            ║
║          Predictor                                   ║
║                                                      ║
║   Advanced ML analyzing 11 critical cardiac params   ║
║                                                      ║
╠═══════════╦═══════════╦═══════════╦══════════════════╣
║    90%    ║   10K+    ║    11     ║      24/7        ║
║ Accuracy  ║Predictions║ Features  ║   Available      ║
╚═══════════╩═══════════╩═══════════╩══════════════════╝
```

</div>

---

## 🧬 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER (11 Features)               │
│  Age · Sex · Chest Pain · BP · Cholesterol · Fasting BS     │
│  Resting ECG · Max HR · Exercise Angina · Oldpeak · Slope   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              ONE-HOT ENCODING (Categorical Features)        │
│  Sex → Sex_M                                                │
│  ChestPainType → ATA / NAP / TA  (ASY = baseline)          │
│  RestingECG → Normal / ST  (LVH = baseline)                 │
│  ExerciseAngina → ExerciseAngina_Y                          │
│  ST_Slope → Flat / Up  (Down = baseline)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STANDARDSCALER  (15 Features Total)            │
│  Normalizes all numerical features to zero mean / unit var  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│     KNeighborsClassifier  (k=5, Minkowski distance)         │
│     Algorithm: auto  ·  Weights: uniform  ·  p=2            │
└──────────────────────────┬──────────────────────────────────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
       ✅ Low Risk (0)          ⚠️ High Risk (1)
```

---

## 📊 Dataset

The model is trained on the **UCI Heart Disease Dataset** — a consolidated dataset combining five major heart disease studies:

| Source | Location | Records |
|---|---|---|
| Cleveland Clinic | Cleveland, USA | ~303 |
| Hungarian Institute | Budapest, Hungary | ~294 |
| V.A. Medical Center | Long Beach, USA | ~200 |
| University Hospital | Zurich, Switzerland | ~123 |
| Statlog (Heart) | — | ~270 |
| **Combined Total** | **5 Sources** | **918** |

### Feature Description

| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | Patient age in years |
| `Sex` | Categorical | M = Male, F = Female |
| `ChestPainType` | Categorical | ATA / NAP / TA / ASY |
| `RestingBP` | Numeric | Resting blood pressure (mm Hg) |
| `Cholesterol` | Numeric | Serum cholesterol (mg/dL) |
| `FastingBS` | Binary | Fasting blood sugar > 120 mg/dL |
| `RestingECG` | Categorical | Normal / ST / LVH |
| `MaxHR` | Numeric | Maximum heart rate achieved |
| `ExerciseAngina` | Categorical | Exercise-induced angina (Y/N) |
| `Oldpeak` | Numeric | ST depression induced by exercise |
| `ST_Slope` | Categorical | Slope of peak exercise ST segment |
| `HeartDisease` | Binary | **Target** — 0 = No, 1 = Yes |

### Target Distribution

```
No Heart Disease (0)  ████████████████████░░░░░░  410  (44.7%)
Heart Disease    (1)  █████████████████████████░  508  (55.3%)
                      Total: 918 records
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/cardioai.git
cd cardioai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📁 Project Structure

```
cardioai/
│
├── 📄 app.py                    # Main Streamlit application
├── 📄 requirements.txt          # Python dependencies
├── 📄 heart.csv                 # Training dataset (918 records)
│
├── 🤖 knn_heart_model.pkl       # Trained KNN classifier
├── ⚖️  heart_scaler.pkl          # Fitted StandardScaler
├── 📋 heart_columns.pkl         # Expected feature column order
│
├── 📓 HeartDiseaseModel.ipynb   # Model training notebook
│
└── 📖 README.md                 # You are here
```

---

## 🔧 Tech Stack

<div align="center">

| Layer | Technology | Version |
|---|---|---|
| **Frontend** | Streamlit | 1.54.0 |
| **ML Framework** | scikit-learn | 1.8.0 |
| **Data Processing** | Pandas + NumPy | 2.3.3 / 2.4.2 |
| **Model Persistence** | joblib | 1.5.3 |
| **Language** | Python | 3.10+ |
| **Visualization** | Altair | 6.0.0 |

</div>

---

## 🧪 Model Performance

```
Algorithm    :  K-Nearest Neighbors (KNN)
K Neighbors  :  5
Distance     :  Minkowski (p=2 = Euclidean)
Weights      :  Uniform
Preprocessing:  StandardScaler (zero mean, unit variance)

Overall Accuracy  ≈  90%
Training Dataset  :  918 patient records
Features Used     :  15 (after one-hot encoding)
```

---

## 📱 App Sections

### 🏠 Home
Overview of CardioAI with key stats, feature highlights, and how the prediction engine works.

### 🔮 Prediction
Interactive form with three sections:
- **Personal Info** — Age, Biological Sex
- **Cardiac Metrics** — Chest pain, BP, Cholesterol, ECG, Max HR
- **Exercise Data** — Angina, ST depression, ST slope

Results are color-coded: 🔴 High Risk or 🟢 Low Risk with actionable guidance.

### 💡 Health Tips
Six evidence-based cardiovascular health cards covering:
Nutrition · Exercise · Smoking · Sleep · Stress Management · Regular Check-ups

### ℹ️ About
Technical details about the ML model, dataset sources, architecture, and full technology stack.

---

## ⚕️ Medical Disclaimer

> CardioAI is built **strictly for educational and research purposes**. The predictions generated by this tool:
> - Are **not a substitute** for professional medical advice
> - Should **not** be used as the sole basis for medical decisions
> - May contain errors and are **not clinically validated**
>
> **Always consult a licensed healthcare professional for cardiac health concerns.**
> In a cardiac emergency, call emergency services immediately — **112 / 911**.

---

## 🤝 Contributing

Contributions are welcome! Here's how:

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
# Open a Pull Request
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with 🫀 and Python**

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
