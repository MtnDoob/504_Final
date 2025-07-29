# ML Movie Final Team Project  

## Project Overview
- **Team Members**: DuyAnh, Saloni, Kirsten
- **Team Number**: [Team #5]
- **Project Title**: Movie Rating Prediction Using Machine Learning

---

## 1. Problem Definition

### Problem Statement
**What is the problem you are going to solve?**
We want to be able to predict movie ratings based on various movie characteristics including content rating, distribution type, descriptions, and other metadata features.

**Why does the problem need to be solved?**
Movie rating prediction is valuable for streaming platforms, movie studios, and recommendation systems to understand audience preferences, optimize content acquisition strategies, and improve user experience. Accurate rating predictions can help stakeholders make informed decisions about movie investments and content curation.

**What aspect of the problem will a machine/deep learning model solve?**
With the abundance of movie metadata available, machine learning models can identify complex patterns and relationships between movie characteristics and audience ratings that would be difficult to detect manually. The models can process multiple features simultaneously to provide accurate rating predictions for new movies.

### STAR Framework Planning
- **Situation**: Movie industry stakeholders need to predict audience reception of movies based on available metadata
- **Task**: Develop and compare multiple machine learning models to predict movie ratings using Letterboxd dataset
- **Action**: Implement comprehensive ML pipeline with preprocessing, feature engineering, and model comparison
- **Result**: Achieve 60.6% variance explanation in movie ratings with Decision Tree as the best-performing model

---

## 2. Dataset Information

### Dataset Details
- **Dataset Name**: Letterboxd
- **Source**: https://www.kaggle.com/datasets/gsimonx37/letterboxd/data
- **Size**: 400,000+ data points (after removing missing values)
- **Features**: 24 processed features (4 numerical, 9 categorical with high-cardinality handling)
- **Target Variable**: Movie ratings (continuous variable)

### Data Quality Assessment
- **Missing Values**: 2,795 rows (0.64%) with missing target values removed
- **Outliers**: Handled through model regularization and tree-based splitting criteria
- **Data Types**: Mixed dataset with numerical features (dates, indices) and categorical features (ratings, types, descriptions)

---

## 3. EDA and Pre-Processing

### Exploratory Data Analysis Plan
-  **Data Overview**
  - Shape: 434,256 rows x 24 columns (after preprocessing)
  - Data types distribution: 4 numerical, 20 categorical (after encoding)
  - Missing value analysis: 0.64% missing in target variable

- **Statistical Summary**
  - Descriptive statistics for numerical features
  - Correlation analysis between features and target
  - Distribution analysis of categorical variables

-  **Visualizations Planned**
  -  Feature importance rankings across models
  -  Model performance comparisons
  -  Distribution of target variable (ratings)

### Pre-Processing Steps
-  **Missing Value Handling**
  - Method: Complete case analysis - removed rows with missing target values
  - Justification: Small percentage (0.64%) of missing data, removal preserves data integrity

-  **Outlier Treatment**
  - Detection method: Handled implicitly through tree-based model splitting criteria
  - Treatment: Retained outliers as they may contain valuable information about extreme cases

-  **Feature Engineering**
  -  High-cardinality categorical encoding using LabelEncoder for features with >50 unique values
  -  One-hot encoding for low-cardinality categorical features
  -  Standard scaling for numerical features

-  **Data Scaling/Normalization**
  - Method: StandardScaler for numerical features
  - Applied to: All numerical features in the preprocessing pipeline

---

## 4. Modeling Methods, Validation, and Performance Metrics

### Models to Implement
-  **Model 1**: Linear Regression
  - Rationale: Baseline model to establish linear relationships
  - Hyperparameters: Default parameters with n_jobs=-1 for parallel processing

-  **Model 2**: Ridge Regression
  - Rationale: Handle potential multicollinearity with L2 regularization
  - Hyperparameters: alpha=1.0, solver='lsqr' for large dataset efficiency

-  **Model 3**: Lasso Regression
  - Rationale: Feature selection through L1 regularization
  - Hyperparameters: alpha=0.1, max_iter=2000

-  **Model 4**: Elastic Net
  - Rationale: Combine L1 and L2 regularization benefits
  - Hyperparameters: alpha=0.1, l1_ratio=0.5, max_iter=2000

-  **Model 5**: Decision Tree
  - Rationale: Capture non-linear relationships and feature interactions
  - Hyperparameters: max_depth=10, min_samples_split=20, min_samples_leaf=10

- **Model 6**: Random Forest
  - Rationale: Ensemble method to reduce overfitting and improve generalization
  - Hyperparameters: n_estimators=30, max_depth=10, min_samples_split=20

-  **Model 7**: Extra Trees
  - Rationale: Alternative ensemble method with random splits
  - Hyperparameters: n_estimators=30, max_depth=10, min_samples_split=20

### Validation Strategy
- **Train/Validation/Test Split**: 80/20 split (347,404 training, 86,852 testing)
- **Cross-Validation**: 3-fold cross-validation for model evaluation
- **Validation Approach**: Hold-out validation with cross-validation for robust performance estimation

### Performance Metrics
**Primary Metrics** (aligned with project objectives):
-  R-squared (R²): Proportion of variance explained in ratings
-  Root Mean Squared Error (RMSE): Magnitude of prediction errors
-  Mean Absolute Error (MAE): Average absolute prediction error

**Secondary Metrics**:
- Cross-validation R² with standard deviation
-  Training vs. Test R² for overfitting assessment

---

## 5. Modeling Results and Findings

### Model Comparison Framework
| Model | Test R² | RMSE | MAE | Training Time | Complexity |
|-------|---------|------|-----|---------------|------------|
| Linear Regression | 0.3927 | 0.4469 | 0.3483 | Fast | Low |
| Ridge Regression | 0.3927 | 0.4469 | 0.3483 | Fast | Low |
| Lasso Regression | 0.1680 | 0.5230 | 0.4199 | Fast | Low |
| Elastic Net | 0.2342 | 0.5018 | 0.3996 | Fast | Low |
| Decision Tree | 0.6060 | 0.3599 | 0.2627 | Medium | Medium |
| Random Forest | 0.5946 | 0.3651 | 0.2798 | High | High |
| Extra Trees | 0.3597 | 0.4588 | 0.3626 | High | High |

### Key Findings
- **Best Performing Model**: Decision Tree (R² = 0.6060) - unexpectedly outperformed ensemble methods
- **Surprising Results**: 
  - Decision Tree outperformed Random Forest, contrary to typical expectations
  - Extra Trees significantly underperformed due to aggressive regularization
  - Ridge and Linear Regression performed identically, suggesting minimal multicollinearity
- **Challenges Encountered**: 
  - Speed optimization required aggressive regularization that hurt Extra Trees performance
  - High-cardinality text features showed low importance with simple label encoding
  - Spurious "Unnamed: 0" feature appeared in importance rankings
- **Feature Importance**: 
  - Most important: rating_y_G (0.38), type_Theatrical (0.20), type_Theatrical limited (0.10)
  - Content rating categories dominate predictive power
  - Distribution types significantly influence ratings
  - Text features (descriptions, taglines) underutilized with current preprocessing

### Business/Practical Implications
The results demonstrate that movie ratings can be predicted with moderate accuracy (60% variance explained) using metadata alone. Content rating (especially G-rated movies) and distribution type are the strongest predictors, suggesting that target audience and release strategy significantly influence ratings. The underperformance of textual features indicates opportunities for improved natural language processing to capture semantic content that could enhance prediction accuracy.

---

## 6. Technical Implementation

### Tools and Libraries
- **Programming Language**: Python
- **Environment**: Google Colab
- **Key Libraries**:
  -  pandas, numpy (data manipulation)
  -  matplotlib, seaborn (visualization)
  -  scikit-learn (ML models, preprocessing, metrics)
  -  scipy (statistical functions)
  -  warnings (error handling)

### Code Organization
- **Data Loading and Cleaning**: MovieRatingsPredictionPipeline.preprocess_data()
-  **EDA**: Integrated within preprocessing and feature importance analysis
-  **Feature Engineering**: ColumnTransformer with StandardScaler and OneHotEncoder
-  **Model Training**: MovieRatingsPredictionPipeline.train_models()
-  **Model Evaluation**: Comprehensive metrics calculation and cross-validation
-  **Results Visualization**: Feature importance analysis and model comparison tables

---

## 7. Report Structure (APA 7 Style)

### Report Outline (7-10 pages)
1. **Introduction**
   - Problem statement: Movie rating prediction using metadata
   - Objectives: Compare ML models for rating prediction accuracy
   - Dataset overview: 434K Letterboxd movie records with 24 features

2. **Literature Review** (if applicable)
   - Related work in movie recommendation systems
   - Methodology justification for ensemble vs. linear methods

3. **Methodology**
   - Data preprocessing: Missing value handling, categorical encoding, scaling
   - Model selection: Seven regression models from linear to ensemble methods
   - Validation approach: 80/20 split with 3-fold cross-validation

4. **Results**
   - Model performance: Decision Tree achieved best R² of 0.6060
   - Comparison analysis: Unexpected Decision Tree superiority over Random Forest
   - Key findings: Content rating and distribution type as primary predictors

5. **Discussion**
   - Interpretation of results: Moderate predictive success with clear feature hierarchy
   - Limitations: Underutilized text features, speed-optimized parameters
   - Future work: Advanced NLP for text features, hyperparameter optimization

6. **Conclusion**
   - Summary of findings: 60% variance explanation with interpretable features
   - Practical implications: Content strategy insights for movie industry

### Appendices
- **Appendix A**: Code (PDF format)
- **Appendix B**: Code (HTML format)
- **Appendix C**: Additional visualizations/tables

---

## 8. Video Presentation Plan

### Presentation Structure (Equal participation)
- **Duration**: 12-15 minutes
- **Team Member 1**: Introduction and Problem Definition (3-4 minutes)
- **Team Member 2**: Methodology and Data Processing (4-5 minutes)
- **Team Member 3**: Results, Findings, and Conclusions (4-5 minutes)

### Presentation Outline
1. **Introduction** (3-4 minutes)
   - Problem statement: Why predict movie ratings?
   - Dataset overview: 434K Letterboxd records

2. **Methodology** (4-5 minutes)
   - EDA highlights: Feature distribution and missing data
   - Model selection rationale: Linear to ensemble comparison

3. **Results** (4-5 minutes)
   - Model comparison: Decision Tree superiority
   - Key findings: Content rating dominance
   - Feature importance visualizations

4. **Conclusion** (2-3 minutes)
   - Summary: 60% predictive accuracy achieved
   - Implications: Content strategy insights
   - Future work: Enhanced text processing

---

## 9. AI Tool Usage Disclosure

### AI Tools Used
-  **Tool 1**: Claude
  - **Purpose**: Code debugging, explanation of results interpretation
  - **Sections**: Pipeline optimization, results analysis
  - **Attribution**: AI assistance noted in code comments

- **Tool 2**: GitHub Copilot (if used)
  - **Purpose**: Code completion and function suggestions
  - **Sections**: Data preprocessing pipeline
  - **Attribution**: Copilot suggestions modified and validated

