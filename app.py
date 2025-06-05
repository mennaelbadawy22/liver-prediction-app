import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Liver Patient Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive-prediction {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .negative-prediction {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> Liver Patient Prediction System</h1>', unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Upload & Training", "Model Prediction", "Data Analysis", "Model Comparison"])

# Helper functions
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the training data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='cp1252')
        else:
            df = pd.read_excel(uploaded_file)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Drop rows with missing target values
        df = df.dropna(subset=["Result"])
        
        # Separate categorical and numerical columns
        cat_cols = df.select_dtypes(include='object').columns
        num_cols = df.select_dtypes(include=['int64', 'float64']).drop(columns="Result").columns
        
        # Handle categorical columns
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            for col in cat_cols:
                df.loc[:, col] = cat_imputer.fit_transform(df[[col]])
                le = LabelEncoder()
                df.loc[:, col] = le.fit_transform(df[col]).astype(int)
        
        # Handle numerical columns
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            df[num_cols] = num_imputer.fit_transform(df[num_cols])
        
        return df, cat_cols, num_cols
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

def train_models(X_train, y_train):
    """Train multiple models and return the best ones"""
    models = {}
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Decision Tree
    status_text.text("Training Decision Tree...")
    dt_params = {'max_depth': [3, 5, 10, None]}
    dt_model = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=3, scoring='f1_weighted')
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model.best_estimator_
    progress_bar.progress(20)
    
    # Random Forest
    status_text.text("Training Random Forest...")
    rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    rf_model = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='f1_weighted')
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model.best_estimator_
    progress_bar.progress(40)
    
    # Logistic Regression
    status_text.text("Training Logistic Regression...")
    lr_params = {'C': [0.1, 1, 10]}
    lr_model = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_params, cv=3, scoring='f1_weighted')
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model.best_estimator_
    progress_bar.progress(60)
    
    # KNN
    status_text.text("Training K-Nearest Neighbors...")
    knn_params = {'n_neighbors': [3, 5, 7]}
    knn_model = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3, scoring='f1_weighted')
    knn_model.fit(X_train, y_train)
    models['KNN'] = knn_model.best_estimator_
    progress_bar.progress(80)
    
    # XGBoost
    status_text.text("Training XGBoost...")
    xgb_params = {'max_depth': [3, 5], 'n_estimators': [50, 100]}
    xgb_model = GridSearchCV(XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42), 
                            xgb_params, cv=3, scoring='f1_weighted')
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model.best_estimator_
    progress_bar.progress(100)
    
    # Soft Voting Ensemble
    status_text.text("Creating Ensemble Model...")
    voting_classifier = VotingClassifier(
        estimators=[
            ('dt', models['Decision Tree']),
            ('rf', models['Random Forest']),
            ('lr', models['Logistic Regression']),
            ('knn', models['KNN']),
            ('xgb', models['XGBoost'])
        ], 
        voting='soft'
    )
    voting_classifier.fit(X_train, y_train)
    models['Ensemble'] = voting_classifier
    
    status_text.text("Training completed!")
    progress_bar.empty()
    status_text.empty()
    
    return models

# Page 1: Data Upload & Training
if page == "Data Upload & Training":
    st.header(" Data Upload & Model Training")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your training data", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        # Load and preprocess data
        with st.spinner("Loading and preprocessing data..."):
            df, cat_cols, num_cols = load_and_preprocess_data(uploaded_file)
        
        if df is not None:
            st.success("Data loaded successfully!")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Features", df.shape[1] - 1)
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Train models button
            if st.button("Train Models", type="primary"):
                with st.spinner("Training models... This may take a few minutes."):
                    # Prepare data
                    X = df.drop('Result', axis=1)
                    y = df['Result'].replace({1.0: 0, 2.0: 1})
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train models
                    models = train_models(X_train_scaled, y_train)
                    
                    # Store in session state
                    st.session_state.models = models
                    st.session_state.scaler = scaler
                    st.session_state.feature_names = X.columns.tolist()
                    st.session_state.X_test = X_test_scaled
                    st.session_state.y_test = y_test
                    st.session_state.model_trained = True
                    
                    st.success(" Models trained successfully!")
                    st.balloons()

# Page 2: Model Prediction
elif page == "Model Prediction":
    st.header("Model Prediction")
    
    if not st.session_state.model_trained:
        st.warning(" Please train the models first in the 'Data Upload & Training' page.")
    else:
        st.success("Models are ready for prediction!")
        
        # Model selection
        model_choice = st.selectbox("Choose a model for prediction", list(st.session_state.models.keys()))
        
        # Feature input
        st.subheader("Enter Patient Information")
        
        # Create input fields based on feature names
        # Note: You'll need to adjust these based on your actual features
        feature_inputs = {}
        
        # Example feature inputs (adjust based on your dataset)
        col1, col2 = st.columns(2)
        
        with col1:
            feature_inputs['Age of the patient'] = st.number_input("Age of the patient", min_value=0, max_value=120, value=40)
            feature_inputs['Gender of the patientr'] = st.selectbox("Gender of the patient", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            feature_inputs['Total_Bilirubin'] = st.number_input("Total Bilirubin", min_value=0.0, value=1.0, step=0.1)
            feature_inputs['Direct_Bilirubin'] = st.number_input("Direct Bilirubin", min_value=0.0, value=0.3, step=0.1)
            feature_inputs['Alkphos Alkaline Phosphotase'] = st.number_input("Alkphos Alkaline Phosphotase", min_value=0, value=200)
          
        
        with col2:
            feature_inputs['Sgpt Alamine Aminotransferase'] = st.number_input("Sgpt Alamine Aminotransferase", min_value=0, value=30)
            feature_inputs['Sgot Aspartate Aminotransferase'] = st.number_input("Sgot Aspartate Aminotransferase", min_value=0, value=30)
            feature_inputs['Total_Protiens'] = st.number_input("Total Proteins", min_value=0.0, value=7.0, step=0.1)
            feature_inputs['ALB Albumin'] = st.number_input("ALB Albumin", min_value=0.0, value=4.0, step=0.1)
            feature_inputs['A/G Ratio Albumin and Globulin Ratio'] = st.number_input("A/G Ratio Albumin and Globulin Ratio", min_value=0.0, value=1.0, step=0.1)
           
        
        # Prediction button
        if st.button("Predict", type="primary"):
            # Prepare input data
            input_data = []
            for feature in st.session_state.feature_names:
                if feature in feature_inputs:
                    input_data.append(feature_inputs[feature])
                else:
                    input_data.append(0)  # Default value for missing features
            
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = st.session_state.scaler.transform(input_array)
            
            # Make prediction
            model = st.session_state.models[model_choice]
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, 'predict_proba') else None
            
            # Display result
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.markdown(
                    '<div class="prediction-result positive-prediction"> High Risk: Patient likely has liver disease</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="prediction-result negative-prediction"> Low Risk: Patient likely does not have liver disease</div>',
                    unsafe_allow_html=True
                )
            
            # Show probability if available
            if prediction_proba is not None:
                st.subheader("Prediction Confidence")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("No Disease Probability", f"{prediction_proba[0]:.2%}")
                with col2:
                    st.metric("Disease Probability", f"{prediction_proba[1]:.2%}")

# Page 3: Data Analysis
elif page == "Data Analysis":
    st.header("📈 Data Analysis & Visualization")
    
    if not st.session_state.model_trained:
        st.warning("Please train the models first to see analysis.")
    else:
        # Feature importance for Random Forest
        if 'Random Forest' in st.session_state.models:
            st.subheader("Feature Importance (Random Forest)")
            rf_model = st.session_state.models['Random Forest']
            importance_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Feature Importance Analysis")
            st.plotly_chart(fig, use_container_width=True)

# Page 4: Model Comparison
elif page == "Model Comparison":
    st.header("📊 Model Performance Comparison")
    
    if not st.session_state.model_trained:
        st.warning(" Please train the models first to see comparison.")
    else:
        # Calculate performance metrics
        performance_data = []
        
        for name, model in st.session_state.models.items():
            y_pred = model.predict(st.session_state.X_test)
            accuracy = accuracy_score(st.session_state.y_test, y_pred)
            
            # Get classification report
            report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
            
            performance_data.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': report['weighted avg']['precision'],
                'Recall': report['weighted avg']['recall'],
                'F1-Score': report['weighted avg']['f1-score']
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Display metrics table
        st.subheader("Performance Metrics")
        st.dataframe(performance_df.round(4))
        
        # Performance comparison chart
        st.subheader("Model Performance Comparison")
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=metrics)
        
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Bar(x=performance_df['Model'], y=performance_df[metric], name=metric),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False, title_text="Model Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model recommendation
        best_model_idx = performance_df['F1-Score'].idxmax()
        best_model = performance_df.loc[best_model_idx, 'Model']
        best_f1 = performance_df.loc[best_model_idx, 'F1-Score']
        
        st.success(f" Best performing model: **{best_model}** with F1-Score of {best_f1:.4f}")

# Footer
st.markdown("---")
st.markdown("Built with using Streamlit | Liver Patient Prediction System")
