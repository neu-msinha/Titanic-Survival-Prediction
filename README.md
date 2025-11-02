# Titanic Survival Prediction

## Course Information
- Course: CSYE 7200 - Big Data Systems Engineering
- Assignment: Spark Assignment 2
- Dataset: Kaggle Titanic Dataset

## Project Overview
This project implements a machine learning pipeline using Apache Spark to predict passenger survival on the Titanic. The solution includes comprehensive exploratory data analysis, feature engineering, and a Random Forest classification model.

## Results

### Model Performance
- Validation Accuracy: 85.52%
- Precision: 85.43%
- Recall: 85.52%
- F1 Score: 85.43%
- AUC-ROC: 0.8985

### Key Findings from EDA
- Overall survival rate: 38.38%
- Female survival rate: 74.20%
- Male survival rate: 18.89%
- First class survival rate: 62.96%
- Second class survival rate: 47.28%
- Third class survival rate: 24.24%

## Technical Stack
- Scala: 2.13.17
- Apache Spark: 3.3.2
- Spark MLlib: Random Forest Classifier
- Build Tool: SBT

## Project Structure
```
Spark-Assignment-2/
├── build.sbt
├── src/
│   └── main/
│       └── scala/
│           └── TitanicSurvivalPrediction.scala
├── train.csv
├── test.csv
├── titanic_submission.csv (generated output)
└── README.md
```

## Features Engineered

### 1. Data Imputation
- Age: Filled with median age by passenger class and gender
- Embarked: Filled with mode (S)
- Fare: Filled with median fare

### 2. Derived Features
- Title: Extracted from passenger names (Mr, Mrs, Miss, Master, Rare)
- FamilySize: Total family members aboard (SibSp + Parch + 1)
- IsAlone: Binary indicator for solo travelers
- AgeGroup: Categorized into Child, Teen, Young_Adult, Adult, Senior
- FareGroup: Categorized into Low, Medium, High, Very_High
- Deck: Extracted from cabin information

### 3. Feature Selection
Dropped unnecessary columns: PassengerId, Name, Ticket, Cabin

## Exploratory Data Analysis

### Analyses Performed
1. Summary statistics of all numerical variables
2. Overall survival rate calculation
3. Survival rate by gender
4. Survival rate by passenger class
5. Age distribution and survival by age group
6. Missing values analysis across all columns
7. Fare statistics by passenger class
8. Family size impact on survival
9. Embarked port survival statistics
10. Combined class and gender survival rates

## Model Details

### Algorithm
Random Forest Classifier with the following parameters:
- Number of trees: 100
- Max depth: 10
- Max bins: 32
- Random seed: 42

### Feature Types
- Categorical Features: Sex, Embarked, Title, AgeGroup, FareGroup, Deck
- Numerical Features: Pclass, Age, SibSp, Parch, Fare, FamilySize, IsAlone

### Pipeline Stages
1. String Indexing for categorical variables
2. One-Hot Encoding
3. Vector Assembly
4. Random Forest Classification

### Training Methodology
- 80/20 train-validation split
- Final model trained on complete training dataset
- Predictions generated on test dataset

## How to Run

### Prerequisites
- Java 17 or higher
- SBT (Scala Build Tool)
- Apache Spark 3.3.2

### Execution Steps

#### Option 1: Using SBT
```bash
cd Spark-Assignment-2
sbt clean
sbt compile
sbt run
```

#### Option 2: Using IntelliJ IDEA
1. Open project in IntelliJ IDEA
2. Configure VM options in Run Configuration:
   - Add VM options for Java 17 compatibility
   - Set "Shorten command line" to "JAR manifest"
3. Run TitanicSurvivalPrediction

### VM Options Required for Java 17+
```
--add-opens=java.base/java.lang=ALL-UNNAMED
--add-opens=java.base/java.lang.invoke=ALL-UNNAMED
--add-opens=java.base/java.lang.reflect=ALL-UNNAMED
--add-opens=java.base/java.io=ALL-UNNAMED
--add-opens=java.base/java.net=ALL-UNNAMED
--add-opens=java.base/java.nio=ALL-UNNAMED
--add-opens=java.base/java.util=ALL-UNNAMED
--add-opens=java.base/java.util.concurrent=ALL-UNNAMED
--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED
--add-opens=java.base/sun.nio.ch=ALL-UNNAMED
--add-opens=java.base/sun.nio.cs=ALL-UNNAMED
--add-opens=java.base/sun.security.action=ALL-UNNAMED
--add-opens=java.base/sun.util.calendar=ALL-UNNAMED
-Xms2G
-Xmx4G
```

## Output

### Console Output
The program generates detailed output including:
- Data loading confirmation
- Complete EDA results
- Feature engineering progress
- Model training status
- Validation metrics
- Prediction statistics

### File Output
- titanic_submission.csv: Contains PassengerId and predicted Survived values for all 418 test passengers

## Assignment Requirements Met

### EDA (20 points)
- 10 comprehensive statistical analyses performed
- Covered survival rates, demographics, missing values, and distributions

### Feature Engineering (30 points)
- 10 features created through transformation and derivation
- Proper handling of missing values
- Categorical encoding and numerical scaling

### Prediction (50 points)
- Model trained exclusively on train.csv
- Survived column used only as label, not as feature
- test.csv used only for predictions, not training
- Achieved 85.52% accuracy (exceeds 70% requirement by 15.52%)

## Model Evaluation

### Confusion Matrix
```
Actual 0, Predicted 0: 81
Actual 0, Predicted 1: 9
Actual 1, Predicted 0: 12
Actual 1, Predicted 1: 43
```

### Interpretation
- True Negatives: 81 (correctly predicted deaths)
- False Positives: 9 (incorrectly predicted survival)
- False Negatives: 12 (incorrectly predicted death)
- True Positives: 43 (correctly predicted survival)

## Dependencies

### build.sbt
```scala
name := "TitanicSurvivalPrediction"
version := "1.0"
scalaVersion := "2.13.17"

val sparkVersion = "3.3.2"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion
)
```

## Contact
Northeastern University
Master's in Software Engineering Systems

## License
Academic Project - CSYE 7200

## Acknowledgments
- Dataset: Kaggle Titanic Competition
- Framework: Apache Spark MLlib
- Course: CSYE 7200 Big Data Systems Engineering
