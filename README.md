# Laptop Price Predictor

A machine learning web app that predicts the price of a laptop based on its
specifications (brand, RAM, CPU, GPU, storage, screen, etc.), built with
Python, Scikit-learn, and Flask.

## Project Structure
```
laptop-price-predictor/
├── app.py                      # Flask web application
├── model.pkl                   # Trained Random Forest model
├── encoders.pkl                # Saved LabelEncoders for categorical features
├── feature_columns.pkl         # Column order expected by the model
├── data/
│   └── laptop_price.csv        # Training dataset
├── notebook/
│   └── Laptop_Price_Prediction.ipynb   # Full data cleaning + training notebook
├── templates/
│   └── index.html              # Web form UI
├── static/
│   └── style.css                # Styling
└── requirements.txt
```

## How It Works
1. **Data Preprocessing** – Load the dataset, drop irrelevant columns, check
   for nulls, explore categorical values.
2. **Encoding** – Convert categorical columns (Company, CPU, GPU, etc.) into
   numeric form using `LabelEncoder`.
3. **Train-Test Split** – 80/20 split for training and evaluation.
4. **Model Training** – Train both `LinearRegression` and
   `RandomForestRegressor`.
5. **Evaluation** – Compare models using R² and MAE; Random Forest performs
   better since laptop pricing is non-linear.
6. **Deployment** – The best model and encoders are saved with `pickle` and
   loaded by a Flask app, where users fill a form and get an instant price
   prediction.

> **Note:** an earlier draft of the training notebook applied `StandardScaler`
> to the target column (`Price_euros`), which meant raw predictions came out
> in scaled units instead of real currency. This version skips scaling
> entirely — Random Forest doesn't need it, and predictions are directly
> usable in euros.

## Running Locally

### Easiest way (Windows) — double-click `run.bat`
Just double-click `run.bat` inside the project folder. It will:
1. Install any missing packages automatically
2. Start the Flask server
3. Open Chrome automatically to the working app

Keep the black terminal window open while using the app — closing it stops the server.

### Manual way (any OS)
```bash
pip install -r requirements.txt

# (Optional) retrain the model
python notebook/train_model.py

# Run the web app
python app.py
```
The app will automatically open in your browser at `http://127.0.0.1:5000`.
If it doesn't open automatically, visit that address manually.

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (LinearRegression, RandomForestRegressor, LabelEncoder)
- Flask (backend/web server)
- HTML/CSS (frontend)
- Pickle (model persistence)

## Model Performance
| Model              | R²     | MAE     |
|--------------------|--------|---------|
| Linear Regression  | ~0.40  | ~377.07 |
| Random Forest       | ~0.84  | ~167.96 |

Random Forest was selected as the final model since it captures the
non-linear relationship between specifications and price more accurately.

## Future Improvements
- Try One-Hot Encoding instead of Label Encoding for nominal features
- Compare against XGBoost / Gradient Boosting
- Add more granular features (battery life, refresh rate)
- Deploy to a cloud platform (Render / Railway / Heroku)
