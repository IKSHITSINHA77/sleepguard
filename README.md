# SleepGuard – Sleep Disorder Prediction & Analytics

**An ML-powered web application for sleep disorder prediction, health assessment, and analytics.**

SleepGuard is a Flask-based machine learning web application designed to help users explore potential sleep-related health risks through a structured assessment. It uses machine learning to predict possible sleep disorder categories and presents prediction results, confidence scores, and analytical insights through a clean, interactive dashboard.

The application includes user registration, secure login, session management, an assessment interface, prediction results, and model analytics.

> **Disclaimer:** SleepGuard is an educational and experimental machine learning project. Its predictions are not medical diagnoses and should not replace professional medical advice or clinical evaluation.

---

## Key Features

### 1. User Authentication & Account Management
- User registration with name, email, and password.
- Secure login and logout functionality.
- Password hashing using Werkzeug security utilities.
- SQLite database for storing user account information.
- Session-based authentication.
- Protected dashboard and API routes.
- CSRF protection for supported forms and prediction requests.
- Personalized dashboard header displaying the logged-in user's name.

### 2. Sleep Disorder Prediction
- Machine learning-based sleep disorder prediction.
- Structured assessment form for entering health and lifestyle information.
- Input validation before generating predictions.
- Prediction categories supported by the trained model:
  - Insomnia
  - Sleep Apnea
  - None
- Confidence score and class probability distribution.
- Clear presentation of prediction results.

### 3. Interactive Dashboard
- Clean and responsive user interface.
- Overview section with project information.
- Dedicated sleep assessment interface.
- Prediction results displayed in a structured format.
- Analytics section for exploring model performance.
- About section explaining the application.

### 4. Machine Learning & Model Analytics
- Model training using Scikit-learn.
- Model evaluation using classification metrics.
- Cross-validation results.
- Confusion matrix visualization.
- Feature importance analysis.
- Classification report.
- Model metadata and performance information.

### 5. Reports & Visualization
- Graphical representation of model analytics.
- Model evaluation reports.
- Feature importance visualization.
- Confusion matrix and classification performance insights.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib |
| Database | SQLite |
| Model Storage | Joblib |
| Authentication | Flask Sessions, Werkzeug Security |
| Testing | Pytest |

---

## Project Structure

```text
SleepGuard/
│
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── data/
│   └── synthetic_sleep.csv    # Synthetic training dataset
│
├── models/
│   └── ...                    # Trained ML models and metadata
│
├── reports/
│   ├── ...                    # Model evaluation reports
│   └── ...                    # Generated visualizations
│
├── scripts/
│   └── ...                    # Supporting scripts
│
├── src/
│   ├── config.py              # Feature configuration and validation
│   ├── train.py               # Model training pipeline
│   ├── predict.py             # Prediction logic
│   └── ...                    # Supporting ML modules
│
├── templates/
│   ├── index.html             # Main dashboard
│   └── auth.html              # Login and registration interface
│
├── tests/
│   └── ...                    # Project tests
│
└── sleepguard_users.db        # SQLite user database (created at runtime)
```

*The exact files inside supporting directories may vary depending on the project version.*

---

## Installation & Setup

### Prerequisites

Make sure you have the following installed:

- Python 3.10 or later
- pip
- Git
- A code editor such as Visual Studio Code

### Step 1: Clone the Repository

```bash
git clone https://github.com/IKSHITSINHA77/SleepGuard.git
```

### Step 2: Navigate to the Project Directory

```bash
cd SleepGuard
```

### Step 3: Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use:

```powershell
.\venv\Scripts\activate.bat
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Train the Machine Learning Model

Run the training module from the project root directory:

```bash
python -m src.train --data data/synthetic_sleep.csv
```

This trains the machine learning model and generates the model artifacts and evaluation outputs required by the application.

> **Dataset note:** The included `synthetic_sleep.csv` contains synthetic data for development and testing. Results obtained from this dataset should not be interpreted as evidence of clinical accuracy or real-world medical performance.

### Step 6: Configure the Flask Secret Key

For stable login sessions across application restarts, set a persistent secret key.

In Windows PowerShell:

```powershell
$env:SLEEPGUARD_SECRET_KEY = "replace-this-with-a-long-random-secret"
```

Keep this key private. Do not commit production secrets to GitHub.

### Step 7: Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000/register
```

Create an account, log in, and access the SleepGuard dashboard.

---

## How to Use SleepGuard

1. Open the application in your browser.
2. Register a new account using your name, email, and password.
3. Log in using your registered credentials.
4. Navigate to the sleep assessment section.
5. Enter the requested health and lifestyle information.
6. Submit the assessment.
7. View the predicted sleep disorder category.
8. Review the confidence score and probability distribution.
9. Explore the analytics dashboard to understand the trained model's performance.
10. Log out securely when finished.

---

## Assessment Features

The prediction pipeline supports the following input fields:

| Feature | Description |
|---|---|
| Gender | User's gender category |
| Age | Age in years |
| Occupation | Occupation category |
| Sleep Duration | Average sleep duration |
| Quality of Sleep | Self-reported sleep quality |
| Physical Activity Level | Physical activity measurement |
| Stress Level | Reported stress level |
| BMI Category | Body mass index category |
| Heart Rate | Heart rate measurement |
| Daily Steps | Number of daily steps |
| Systolic | Systolic blood pressure |
| Diastolic | Diastolic blood pressure |

The application validates the submitted information before passing it to the trained prediction model.

---

## Machine Learning Workflow

SleepGuard follows a structured machine learning workflow:

```text
Dataset
   |
   v
Data Preprocessing
   |
   v
Feature Preparation
   |
   v
Model Training
   |
   v
Model Evaluation
   |
   v
Model Selection
   |
   v
Model Saving
   |
   v
Prediction through Flask API
   |
   v
Results on Dashboard
```

### Model Evaluation

The project includes analytical outputs such as:

- Accuracy and classification metrics.
- Macro F1-score.
- Cross-validation results.
- Confusion matrix.
- Classification report.
- Feature importance.

These metrics help evaluate the trained model on the available dataset.

---

## API Endpoints

The Flask application provides the following routes:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/register` | Display registration page |
| POST | `/register` | Register a new user |
| GET | `/login` | Display login page |
| POST | `/login` | Authenticate user |
| POST | `/logout` | Log out the current user |
| GET | `/` | Access the protected dashboard |
| POST | `/api/predict` | Generate a sleep disorder prediction |
| GET | `/api/analytics` | Retrieve model analytics |
| GET | `/reports/<filename>` | Access generated reports |
| GET | `/health` | Application health check |

**Authentication:** Dashboard, prediction, analytics, and report routes are protected and require an authenticated session.

**CSRF protection:** Supported form submissions and prediction requests require a valid CSRF token.

---

## Security Features

SleepGuard incorporates basic web application security practices:

- Passwords are stored as secure hashes rather than plain text.
- Authenticated sessions protect restricted application routes.
- HTTP-only session cookies.
- SameSite cookie configuration.
- CSRF protection for supported requests.
- Configurable secure-cookie behavior for HTTPS deployments.
- Environment-based Flask secret key configuration.

For production deployment, additional security measures are recommended, including HTTPS, login rate limiting, email verification, password recovery, and a production-grade database configuration.

---

## Important Notes

- The application is designed for educational and demonstration purposes.
- The included synthetic dataset is intended for development and experimentation.
- Prediction confidence is a model output, not a measure of clinical certainty.
- The SQLite database stores user account information locally.
- User-specific prediction history and saved reports are not currently implemented.
- The Flask development server should not be used as a production hosting server.
- Set a persistent `SLEEPGUARD_SECRET_KEY` before using the application across restarts.

---

## Future Enhancements

Potential improvements include:

- Individual user prediction history.
- Personalized sleep health reports.
- Downloadable PDF assessment reports.
- Advanced machine learning and deep learning models.
- Integration with wearable devices and sleep tracking systems.
- Improved model explainability using SHAP.
- Email verification and password reset.
- Production database integration.
- Cloud deployment and monitoring.
- Enhanced data security and access controls.

---

## Project Objective

The objective of SleepGuard is to demonstrate how machine learning, backend development, data analytics, and web technologies can be combined to create an interactive application for exploring sleep-related health patterns.

The project provides practical experience in:

- Machine learning model development.
- Data preprocessing and model evaluation.
- Python backend development with Flask.
- REST API integration.
- User authentication and session management.
- SQLite database integration.
- Frontend and backend communication.
- Data visualization and analytics.

---

## Author

**Ikshit Sinha**

B.Tech – Computer Science Engineering (Cybersecurity)  
Dayananda Sagar Academy of Technology and Management, Bengaluru

**Areas of Interest:** Artificial Intelligence, Machine Learning, Generative AI, Python, and Backend Development.

- GitHub: https://github.com/IKSHITSINHA77
- LinkedIn: https://linkedin.com/in/ikshit-sinha-a76883288/

---

## License

This project is intended for educational and learning purposes. Add an appropriate open-source license to the repository if you intend to permit public reuse and distribution.

---

**Built with Python, Flask, Machine Learning, and a focus on better sleep-health insights.**
