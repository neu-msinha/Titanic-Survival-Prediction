import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.{Pipeline, PipelineModel}

object TitanicSurvivalPrediction  {

  case class ModelMetrics(accuracy: Double, precision: Double, recall: Double, f1: Double, auc: Double)
  case class PredictionResult(survived: Long, died: Long, total: Long)

  def main(args: Array[String]): Unit = {
    val spark = createSparkSession()
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._

    printHeader("TITANIC SURVIVAL PREDICTION - CSYE 7200 ASSIGNMENT")

    val result = for {
      data <- loadData(spark)
      _ = displayDataInfo(data._1, data._2)
      _ = performEDA(data._1, spark)
      processed <- processFeatures(data._1, data._2, spark)
      model <- trainAndEvaluate(processed._1, spark)
      predictions = generatePredictions(model._1, processed._2, spark)
      _ <- savePredictions(predictions, spark)
    } yield model._2

    result.foreach(printSummary)
    spark.stop()
  }

  def createSparkSession(): SparkSession = {
    SparkSession.builder()
      .appName("Titanic Survival Prediction - CSYE 7200")
      .master("local[*]")
      .config("spark.driver.memory", "4g")
      .getOrCreate()
  }

  def loadData(spark: SparkSession): Option[(DataFrame, DataFrame)] = {
    try {
      val trainDf = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("train.csv")
        .cache()

      val testDf = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("test.csv")
        .cache()

      Some((trainDf, testDf))
    } catch {
      case e: Exception =>
        println(s"Error loading data: ${e.getMessage}")
        None
    }
  }

  def featureEngineering(
                          df: DataFrame,
                          referenceDf: DataFrame,
                          spark: SparkSession,
                          keepPassengerId: Boolean
                        ): DataFrame = {
    import spark.implicits._

    val medianAges = referenceDf.groupBy("Pclass", "Sex")
      .agg(expr("percentile_approx(Age, 0.5)").alias("MedianAge"))

    val modeEmbarked = referenceDf
      .filter($"Embarked".isNotNull)
      .groupBy("Embarked")
      .count()
      .orderBy(desc("count"))
      .first()
      .getString(0)

    val medianFare = referenceDf.stat.approxQuantile("Fare", Array(0.5), 0.01)(0)

    val rareTitles = Set("Lady", "Countess", "Capt", "Col", "Don",
      "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona")

    val result = df
      .join(medianAges, Seq("Pclass", "Sex"), "left")
      .withColumn("Age", when($"Age".isNull, $"MedianAge").otherwise($"Age"))
      .drop("MedianAge")
      .withColumn("Embarked",
        when($"Embarked".isNull || $"Embarked" === "", lit(modeEmbarked))
          .otherwise($"Embarked"))
      .withColumn("Fare",
        when($"Fare".isNull, lit(medianFare)).otherwise($"Fare"))
      .withColumn("Title", regexp_extract($"Name", " ([A-Za-z]+)\\.", 1))
      .withColumn("Title",
        when($"Title".isin(rareTitles.toSeq: _*), "Rare")
          .when($"Title" === "Mlle", "Miss")
          .when($"Title" === "Ms", "Miss")
          .when($"Title" === "Mme", "Mrs")
          .otherwise($"Title"))
      .withColumn("FamilySize", $"SibSp" + $"Parch" + 1)
      .withColumn("IsAlone", when($"FamilySize" === 1, 1).otherwise(0))
      .withColumn("AgeGroup",
        when($"Age" <= 12, "Child")
          .when($"Age" <= 18, "Teen")
          .when($"Age" <= 35, "Young_Adult")
          .when($"Age" <= 60, "Adult")
          .otherwise("Senior"))
      .withColumn("FareGroup",
        when($"Fare" <= 7.91, "Low")
          .when($"Fare" <= 14.454, "Medium")
          .when($"Fare" <= 31, "High")
          .otherwise("Very_High"))
      .withColumn("Deck",
        when($"Cabin".isNull || $"Cabin" === "", "Unknown")
          .otherwise(substring($"Cabin", 1, 1)))
      .transform { df =>
        val columnsToDrop = if (keepPassengerId) {
          List("Name", "Ticket", "Cabin")
        } else {
          List("PassengerId", "Name", "Ticket", "Cabin")
        }
        df.drop(columnsToDrop: _*)
      }

    result
  }

  def processFeatures(
                       trainDf: DataFrame,
                       testDf: DataFrame,
                       spark: SparkSession
                     ): Option[(DataFrame, DataFrame)] = {
    try {
      printHeader("STEP 3: FEATURE ENGINEERING")

      val trainProcessed = featureEngineering(trainDf, trainDf, spark, keepPassengerId = false)
      val testProcessed = featureEngineering(testDf, trainDf, spark, keepPassengerId = true)

      displayFeatureInfo(trainProcessed)

      Some((trainProcessed, testProcessed))
    } catch {
      case e: Exception =>
        println(s"Error in feature engineering: ${e.getMessage}")
        None
    }
  }

  def createPipeline(
                      categoricalColumns: List[String],
                      numericalColumns: List[String]
                    ): Pipeline = {

    val indexers = categoricalColumns.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(s"${colName}_index")
        .setHandleInvalid("keep")
    }

    val encoders = categoricalColumns.map { colName =>
      new OneHotEncoder()
        .setInputCol(s"${colName}_index")
        .setOutputCol(s"${colName}_vec")
    }

    val featureColumns = categoricalColumns.map(col => s"${col}_vec") ++ numericalColumns
    val assembler = new VectorAssembler()
      .setInputCols(featureColumns.toArray)
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val rf = new RandomForestClassifier()
      .setLabelCol("Survived")
      .setFeaturesCol("features")
      .setNumTrees(100)
      .setMaxDepth(10)
      .setMaxBins(32)
      .setSeed(42)

    new Pipeline().setStages((indexers ++ encoders :+ assembler :+ rf).toArray)
  }

  def evaluateModel(predictions: DataFrame): ModelMetrics = {
    val evaluatorAcc = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val evaluatorPrecision = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("prediction")
      .setMetricName("weightedPrecision")

    val evaluatorRecall = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("prediction")
      .setMetricName("weightedRecall")

    val evaluatorF1 = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val evaluatorAUC = new BinaryClassificationEvaluator()
      .setLabelCol("Survived")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    ModelMetrics(
      accuracy = evaluatorAcc.evaluate(predictions),
      precision = evaluatorPrecision.evaluate(predictions),
      recall = evaluatorRecall.evaluate(predictions),
      f1 = evaluatorF1.evaluate(predictions),
      auc = evaluatorAUC.evaluate(predictions)
    )
  }

  def trainAndEvaluate(df: DataFrame, spark: SparkSession): Option[(PipelineModel, ModelMetrics)] = {
    import spark.implicits._

    val categoricalColumns = List("Sex", "Embarked", "Title", "AgeGroup", "FareGroup", "Deck")
    val numericalColumns = List("Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone")

    val pipeline = createPipeline(categoricalColumns, numericalColumns)
    val Array(trainData, validationData) = df.randomSplit(Array(0.8, 0.2), seed = 42)

    try {
      printHeader("STEP 4: MODEL TRAINING AND EVALUATION")

      val model = pipeline.fit(trainData)
      val predictions = model.transform(validationData)
      val metrics = evaluateModel(predictions)

      displayTrainingResults(trainData, validationData, metrics, predictions)

      val finalModel = pipeline.fit(df)

      Some((finalModel, metrics))
    } catch {
      case e: Exception =>
        println(s"Error during training: ${e.getMessage}")
        None
    }
  }

  def generatePredictions(model: PipelineModel, testProcessed: DataFrame, spark: SparkSession): DataFrame = {
    import spark.implicits._

    val predictions = model.transform(testProcessed)
    displayPredictionStats(predictions)
    predictions
  }

  def savePredictions(predictions: DataFrame, spark: SparkSession): Option[String] = {
    import spark.implicits._

    val submission = predictions
      .select($"PassengerId", $"prediction".cast("int").alias("Survived"))
      .orderBy("PassengerId")

    printHeader("SAVING RESULTS")
    println("\nSubmission Preview:")
    submission.show(20)

    try {
      import java.io.PrintWriter
      val writer = new PrintWriter("titanic_submission.csv")
      writer.println("PassengerId,Survived")

      submission.collect().foreach { row =>
        writer.println(s"${row.getInt(0)},${row.getInt(1)}")
      }

      writer.close()
      println("\nResults saved to: titanic_submission.csv")
      Some("titanic_submission.csv")
    } catch {
      case e: Exception =>
        println(s"Error saving file: ${e.getMessage}")
        None
    }
  }

  // EDA
  def performEDA(df: DataFrame, spark: SparkSession): Unit = {
    import spark.implicits._

    printHeader("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    println("Following up on Assignment 1 with additional insights\n")

    val total = df.count().toDouble

    // ASSIGNMENT 1 FOLLOW-UP QUESTIONS

    println("\n" + "-"*70)
    println("ASSIGNMENT 1 FOLLOW-UP ANALYSIS")
    println("-"*70)

    // 1. Average Fare by Class (Assignment 1 Question 1)
    println("\n[Assignment 1 Q1] AVERAGE TICKET FARE BY CLASS")
    println("-" * 70)
    val fareByClass = df.groupBy("Pclass")
      .agg(
        avg("Fare").alias("Avg_Fare"),
        min("Fare").alias("Min_Fare"),
        max("Fare").alias("Max_Fare"),
        stddev("Fare").alias("StdDev_Fare")
      )
      .orderBy("Pclass")
    fareByClass.show()

    // 2. Survival Percentage by Class (Assignment 1 Question 2)
    println("\n[Assignment 1 Q2] SURVIVAL PERCENTAGE BY CLASS")
    println("-" * 70)
    val survivalByClass = df.groupBy("Pclass")
      .agg(
        count("*").alias("Total"),
        sum("Survived").alias("Survived")
      )
      .withColumn("Survival_Percentage", round($"Survived" / $"Total" * 100, 2))
      .orderBy("Pclass")
    survivalByClass.show()

    val highestSurvivalClass = survivalByClass
      .orderBy(desc("Survival_Percentage"))
      .first()
    println(s"Highest survival rate: Class ${highestSurvivalClass.getInt(0)} with ${highestSurvivalClass.getDouble(3)}%")

    // 3. Find Rose DeWitt Bukater (Assignment 1 Question 3)
    println("\n[Assignment 1 Q3] PASSENGERS MATCHING ROSE DEWITT BUKATER")
    println("-" * 70)
    println("Rose: 17 years old, female, 1st class, traveling with 1 parent (mother)")
    val possibleRose = df.filter(
      $"Age" === 17 &&
        $"Sex" === "female" &&
        $"Pclass" === 1 &&
        $"Parch" === 1
    )
    println(s"Number of passengers matching Rose: ${possibleRose.count()}")
    possibleRose.select("PassengerId", "Name", "Age", "Sex", "Pclass", "Parch", "Survived").show(false)

    // 4. Find Jack Dawson (Assignment 1 Question 4)
    println("\n[Assignment 1 Q4] PASSENGERS MATCHING JACK DAWSON")
    println("-" * 70)
    println("Jack: 19-20 years old, male, 3rd class, no relatives onboard")
    val possibleJack = df.filter(
      ($"Age" === 19 || $"Age" === 20) &&
        $"Sex" === "male" &&
        $"Pclass" === 3 &&
        $"SibSp" === 0 &&
        $"Parch" === 0
    )
    println(s"Number of passengers matching Jack: ${possibleJack.count()}")
    possibleJack.select("PassengerId", "Name", "Age", "Sex", "Pclass", "SibSp", "Parch", "Survived").show(false)

    // 5. Age Groups Analysis (Assignment 1 Question 5)
    println("\n[Assignment 1 Q5] AGE GROUPS (10-YEAR INTERVALS) VS FARE & SURVIVAL")
    println("-" * 70)
    val ageGroupAnalysis = df.filter($"Age".isNotNull)
      .withColumn("AgeGroup",
        concat(
          floor($"Age" / 10) * 10 + 1,
          lit("-"),
          floor($"Age" / 10) * 10 + 10
        )
      )
      .groupBy("AgeGroup")
      .agg(
        count("*").alias("Count"),
        avg("Fare").alias("Avg_Fare"),
        avg("Survived").alias("Survival_Rate")
      )
      .withColumn("Survival_Percentage", round($"Survival_Rate" * 100, 2))
      .orderBy("AgeGroup")

    ageGroupAnalysis.show()

    val highestSurvivalAge = ageGroupAnalysis
      .orderBy(desc("Survival_Rate"))
      .first()
    println(s"Age group most likely to survive: ${highestSurvivalAge.getString(0)} with ${highestSurvivalAge.getDouble(4)}% survival rate")

    // ADDITIONAL EDA FOR ASSIGNMENT 2

    println("\n" + "-"*70)
    println("ADDITIONAL EXPLORATORY ANALYSIS")
    println("-"*70)

    // 6. Overall Survival Rate
    println("\n[6] OVERALL SURVIVAL RATE")
    val survived = df.filter($"Survived" === 1).count()
    println(s"Total: ${total.toInt}, Survived: $survived (${(survived/total * 100).formatted("%.2f")}%)")
    df.groupBy("Survived").count().show()

    // 7. Survival by Gender
    println("\n[7] SURVIVAL BY GENDER")
    val genderStats = df.groupBy("Sex")
      .agg(count("*").alias("Total"), sum("Survived").alias("Survived"), avg("Survived").alias("Rate"))
      .withColumn("Survival_Pct", round($"Rate" * 100, 2))
    genderStats.show()

    // 8. Missing Values
    println("\n[8] MISSING VALUES ANALYSIS")
    df.columns.foreach { col =>
      val nulls = df.filter(df(col).isNull || df(col) === "").count()
      println(f"$col%-15s: $nulls%4d (${nulls/total * 100}%.2f%%)")
    }

    // 9. Family Size Impact
    println("\n[9] FAMILY SIZE ANALYSIS")
    df.withColumn("FamilySize", $"SibSp" + $"Parch" + 1)
      .groupBy("FamilySize")
      .agg(count("*").alias("Count"), avg("Survived").alias("Rate"))
      .withColumn("Survival_Pct", round($"Rate" * 100, 2))
      .orderBy("FamilySize")
      .show()

    // 10. Embarked Port
    println("\n[10] EMBARKED PORT STATISTICS")
    df.groupBy("Embarked")
      .agg(count("*").alias("Total"), avg("Survived").alias("Rate"))
      .withColumn("Survival_Pct", round($"Rate" * 100, 2))
      .show()

    println("\nEDA Complete - Followed up on all Assignment 1 questions!")
  }
  def printHeader(title: String): Unit = {
    println("\n" + "="*70)
    println(title)
    println("="*70)
  }

  def displayDataInfo(train: DataFrame, test: DataFrame): Unit = {
    println(s"\n[STEP 1] Loading Datasets...")
    println(s"Train Dataset: ${train.count()} rows, ${train.columns.length} columns")
    println(s"Test Dataset: ${test.count()} rows, ${test.columns.length} columns")
    println("\nTrain Dataset Schema:")
    train.printSchema()
    println("\nFirst 5 rows:")
    train.show(5, truncate = false)
  }

  def displayFeatureInfo(df: DataFrame): Unit = {
    println("\n[1] Age: Filled with median by class and gender")
    println("[2] Embarked: Filled with mode")
    println("[3] Fare: Filled with median")
    println("[4] Title: Extracted and grouped")
    println("[5] FamilySize: Created from SibSp + Parch + 1")
    println("[6] IsAlone: Binary indicator")
    println("[7] AgeGroup: Categorized into 5 groups")
    println("[8] FareGroup: Categorized into 4 groups")
    println("[9] Deck: Extracted from cabin")
    println("[10] Dropped: PassengerId, Name, Ticket, Cabin")
    println(s"\nFinal Features: ${df.columns.mkString(", ")}")
    println("\nSample Data:")
    df.show(5, truncate = false)
  }

  def displayEDAResults(
                         df: DataFrame,
                         total: Double,
                         survivalStats: (Long, Long),
                         genderStats: DataFrame,
                         classStats: DataFrame,
                         ageGroupStats: DataFrame,
                         missingValues: Map[String, Long],
                         fareStats: DataFrame,
                         familyStats: DataFrame,
                         embarkedStats: DataFrame,
                         combinedStats: DataFrame
                       ): Unit = {

    println("\n[1] SUMMARY STATISTICS")
    println("-" * 70)
    df.describe().show()

    println("\n[2] OVERALL SURVIVAL RATE")
    println("-" * 70)
    println(s"Total: ${total.toInt}, Survived: ${survivalStats._1}, Died: ${survivalStats._2}")
    println(f"Survival Rate: ${survivalStats._1/total * 100}%.2f%%")

    println("\n[3] SURVIVAL BY GENDER")
    println("-" * 70)
    genderStats.show()

    println("\n[4] SURVIVAL BY PASSENGER CLASS")
    println("-" * 70)
    classStats.orderBy("Pclass").show()

    println("\n[5] AGE DISTRIBUTION")
    println("-" * 70)
    df.select("Age").describe().show()
    ageGroupStats.show()

    println("\n[6] MISSING VALUES ANALYSIS")
    println("-" * 70)
    missingValues.foreach { case (col, count) =>
      println(f"$col%-15s: $count%4d (${count/total * 100}%.2f%%)")
    }

    println("\n[7] FARE STATISTICS BY CLASS")
    println("-" * 70)
    fareStats.orderBy("Pclass").show()

    println("\n[8] FAMILY SIZE ANALYSIS")
    println("-" * 70)
    familyStats.orderBy("FamilySize").show()

    println("\n[9] EMBARKED PORT STATISTICS")
    println("-" * 70)
    embarkedStats.show()

    println("\n[10] CLASS & GENDER COMBINED")
    println("-" * 70)
    combinedStats.orderBy("Pclass", "Sex").show()

    println("\nEDA Complete!")
  }

  def displayTrainingResults(
                              train: DataFrame,
                              validation: DataFrame,
                              metrics: ModelMetrics,
                              predictions: DataFrame
                            ): Unit = {
    println(s"\n[1] Data Split")
    println(s"   Training Set: ${train.count()} records")
    println(s"   Validation Set: ${validation.count()} records")

    println("\n[2] MODEL EVALUATION RESULTS")
    println("-" * 70)
    println(f"   Accuracy:  ${metrics.accuracy * 100}%.2f%%")
    println(f"   Precision: ${metrics.precision * 100}%.2f%%")
    println(f"   Recall:    ${metrics.recall * 100}%.2f%%")
    println(f"   F1 Score:  ${metrics.f1 * 100}%.2f%%")
    println(f"   AUC-ROC:   ${metrics.auc}%.4f")

    println("\n[3] CONFUSION MATRIX")
    println("-" * 70)
    predictions.groupBy("Survived", "prediction").count()
      .orderBy("Survived", "prediction").show()
  }

  def displayPredictionStats(predictions: DataFrame): Unit = {
    import predictions.sparkSession.implicits._

    val stats = PredictionResult(
      survived = predictions.filter($"prediction" === 1.0).count(),
      died = predictions.filter($"prediction" === 0.0).count(),
      total = predictions.count()
    )

    printHeader("GENERATING PREDICTIONS ON TEST SET")
    println(s"\nPredicted Survived: ${stats.survived}")
    println(s"Predicted Died: ${stats.died}")
    println(s"Total: ${stats.total} \n")
  }

  def printSummary(metrics: ModelMetrics): Unit = {
    println(f"Model Validation Accuracy: ${metrics.accuracy * 100}%.2f%%")
    println("Predictions saved to: titanic_submission.csv")
    println("\nEDA: 10 analyses (20 pts)")
    println("Feature Engineering: 10 features (30 pts)")
    println(f"Model Accuracy: ${metrics.accuracy * 100}%.2f%% (50 pts)")
    println("="*70)
  }
}