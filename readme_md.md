# 🏥 Liver Patient Prediction System

A comprehensive machine learning web application for predicting liver disease using multiple classification algorithms.

## 🚀 Live Demo

[Deploy your app and add the link here]

## 📋 Features

- **Data Upload & Processing**: Upload CSV/Excel files with automatic preprocessing
- **Multiple ML Models**: Decision Tree, Random Forest, Logistic Regression, KNN, XGBoost, and Ensemble
- **Interactive Predictions**: Real-time patient risk assessment
- **Data Analysis**: Feature importance and correlation analysis
- **Model Comparison**: Performance metrics visualization
- **User-Friendly Interface**: Modern, responsive web design

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deployment**: Streamlit Community Cloud

## 📊 Models Implemented

1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **Logistic Regression**
4. **K-Nearest Neighbors (KNN)**
5. **XGBoost Classifier**
6. **Soft Voting Ensemble**

## 🏃‍♂️ Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/liver-prediction-app.git
   cd liver-prediction-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

### Usage

1. **Upload Data**: Go to "Data Upload & Training" and upload your liver patient dataset
2. **Train Models**: Click "Train Models" to train all 6 machine learning models
3. **Make Predictions**: Navigate to "Model Prediction" to assess individual patients
4. **Analyze Results**: Use "Data Analysis" and "Model Comparison" for insights

## 📁 Project Structure

```
liver-prediction-app/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore file
└── sample_data/          # Sample datasets (optional)
    └── sample_liver_data.csv
```

## 🔧 Data Format

Your training data should include the following columns:
- Age
- Total_Bilirubin
- Direct_Bilirubin
- Alkaline_Phosphotase
- Alamine_Aminotransferase
- Aspartate_Aminotransferase
- Total_Protiens
- Albumin
- Albumin_and_Globulin_Ratio
- Gender (0 for Female, 1 for Male)
- Result (1 for No Disease, 2 for Disease)

## 📈 Model Performance

The application provides comprehensive performance metrics including:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC curves
- Confusion matrices

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: [Add your data source here]
- Built with Streamlit and Scikit-learn
- Inspired by healthcare ML applications

## 📞 Contact

Your Name - [your.email@example.com]

Project Link: [https://github.com/yourusername/liver-prediction-app](https://github.com/yourusername/liver-prediction-app)

---

⭐ **Star this repository if you found it helpful!**