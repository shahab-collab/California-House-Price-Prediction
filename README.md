# 🏡 California Housing Price Prediction
This is my first hands-on ML project focusing on learning the basic concepts and libraries through hands-on practice with personal documentation.
I also made a live demo of the model and created a simple UI with Gradio and deployed it on HuggingFace Spaces, you can check it out.
I explained everything in the README file.

## 📌 Project Purpose (short)
This project was built as my first hands-on machine learning project: to learn the end-to-end workflow 
(data exploration, cleaning, feature engineering, model selection,tuning, evaluation and deployment). 
It is primarily a learning / demo project rather than a production-grade system.
the model still has room for improvement — but it demonstrates an end-to-end pipeline and a live demo hosted with **Gradio** on Hugging Face Spaces.

---

## 🚀 Live demo
**Live demo (Hugging Face Space):** [https://huggingface.co/spaces/shahab308/california-housing-predictor](https://huggingface.co/spaces/shahab308/california-housing-predictor)

---

## 📊 Quick model summary
- **Dataset:** California housing dataset.  
- **Target:** `median_house_value` (median house price).  
- **Model class:** Scikit-learn estimator (trained inside a preprocessing pipeline).  
- **Evaluation:** RMSE ≈ **$48,000**, roughly **~20–25%** of a typical house value in the dataset — reasonable for a first end-to-end exercise, but not production quality.

---

## 🎯 Motivation & goals
My main goals were to:

- Get practical experience building a full ML workflow (not only model code).  
- Learn the scikit-learn pipeline mechanisms, custom transformers, and how to save/load a model (using `joblib`).  
- Practice EDA, data cleaning, feature engineering, and model evaluation rather than just fitting a single algorithm.  
- Learn basic deployment (Gradio + Hugging Face Spaces) and debug the entire flow on a remote server.  
- Familiarize myself with tools: `numpy`, `pandas`, `scikit-learn`, and joblib (and get an intro to `matplotlib`/`scipy` while exploring).

---

## 🔬 What I actually did — step by step
Below I describe the workflow I followed (similar to the step-by-step approach in the *Hands-On Machine Learning* book).

### 1. Exploration (EDA)
- Inspected the dataset for shape, types, and basic statistics.  
- Plotted distributions and relationships to find obvious patterns and outliers.  
- Checked for missing values — `total_bedrooms` had missing entries that needed handling.

### 2. Handling imbalanced target / sampling
- Observed that the target / some predictors were not uniformly distributed.  
- Used `pd.cut` to create income categories and then `StratifiedShuffleSplit` to create a stratified train/test split.
  this prevents sampling bias and keeps the train/test distribution consistent.

### 3. Missing values
- Used a `SimpleImputer` with **median** strategy to fill missing numeric values (robust to outliers), especially for `total_bedrooms`.

### 4. Categorical feature
- The CSV version of this dataset contains a categorical feature: `ocean_proximity`.  
- Applied one-hot encoding (inside the pipeline) so the model can use this information.

### 5. Feature engineering (custom transformer)
- Implemented a small custom transformer to create useful derived features, e.g.:
  - `rooms_per_household = total_rooms / households`
  - `population_per_household = population / households`
  - `bedrooms_per_room = total_bedrooms / total_rooms` (optional)
- Wrapped these steps and the imputer/scaler/encoder into a reproducible `Pipeline` (so final saved object contains both preprocessing and the estimator).

> Note: Because the pipeline included a custom transformer, the same transformer class definition must be available in the deployment script
  when loading the saved `.pkl`. (I realized this when I got a runtime error)

### 6. Scaling and encoding
- Applied `StandardScaler` to numeric features and `OneHotEncoder` to categorical feature (ocean proximity) - all as part of the pipeline.

### 7. Model selection & tuning
- Tried multiple models (namely linear regression, decision tree regression, random forest regressor) and compared them with cross-validation.  
- Performed hyperparameter tuning (grid search / randomized search) to find a reasonable configuration.  
- Kept cross-validation scores (and their mean/std) to compare model stability.

### 8. Final evaluation and saving
- Selected a final estimator and retrained on the training set.  
- Evaluated on the hold-out test set to get the final RMSE (~$48k).  
- Saved the full pipeline (preprocessing + final estimator) using `joblib.dump(full_model, "full_pipeline_model.pkl")` to `full_pipeline_model.pkl` for deployment.

### 9. Deployment
- Built a small Gradio app (`app.py`) that accepts human-friendly inputs, fills fixed defaults for geographic features (longitude/latitude),
  constructs a single-row `DataFrame` with the exact column names expected by the pipeline, calls `full_pipeline_model.predict(...)`, and returns a formatted string.  
- Hosted the app on **Hugging Face Spaces** for a quick demo. Gradio was chosen for speed of deployment
  and the automatic UI (instead of building a FastAPI frontendfrom scratch).

---

## ⚠️ Challenges & lessons learned
- **Custom transformer persistence:** When you save a pipeline that uses custom Python classes, you must provide the same class definition
   in the environment that loads the pipeline (or re-save the model using standard transformers only). I ran into `AttributeError` during
   unpickling and fixed it by including the transformer code in the deployment file.
- **Version mismatch warnings:** Scikit-learn versions used for saving and loading may differ; this raises `InconsistentVersionWarning`.
   For robust deployment pin versions when necessary (e.g., `scikit-learn==1.6.1`).
- **Data pipeline correctness:** Deployment bugs often stem from mismatched input column names or types. Building the `DataFrame` with
   exact column names is crucial.
- **Monitoring & maintainability:** I briefly considered model monitoring strategies (drift detection, retraining schedule) to avoid model rot
   — not implemented in this demo, but an important future step.
- **Practicality vs. speed:** I chose Gradio for a fast demo. For production, a dedicated FastAPI app plus logging/monitoring and authentication would be better.

---

## ⚖️ Limitations
- The model’s RMSE (~$48k) means predictions are noisy — acceptable for learning/demos, but not production-ready for pricing decisions.  
- No live monitoring, no input validation beyond basic checks, and no authentication — all of these would be required for a production deployment.  
- The model and pipeline were trained on historical data; distributional drift over time is not addressed.

---

## 🔭 Future work (next steps)
- Add monitoring (data drift, performance tracking) and retraining pipelines.  
- Improve feature engineering (spatial features, neighborhood aggregation) and test more algorithms (e.g., gradient boosting).  
- Add stronger input validation and user guidance in the UI (ranges, example values).  
- Build a clean FastAPI backend with a separate frontend (React) if a production UI is needed.
- Containerize the app with a pinned runtime (Docker) and use a CI/CD pipeline for safe updates.

---

## 🛠️ Tools & libs used (high level)
- `numpy`, `pandas` — data handling  
- `scikit-learn` — preprocessing, pipeline, models, CV  
- `joblib` — saving/loading pipeline  
- `gradio` — demo UI  
- Matplotlib — exploratory visualizations (used in notebooks)

---

## 📚 Acknowledgements
- Much of the workflow and educational guidance follows the approach in *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (A. Géron).
  I used the book as a learning guide and adapted examples for practice.

---
## 📞 Contact / notes
- If you try the demo and want the source code or have questions, open an issue or reach out on LinkedIn / GitHub (links in my profile).  
- This project is licensed under the **MIT License**.
