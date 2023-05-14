package org.apache.spark.ml.made


import org.apache.spark.sql.functions.udf
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.made.LogisticRegressionTest.{sqlc}
import org.apache.spark.sql.functions.{lit, udf}
import sqlc.implicits._

class LogisticRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.01
  val expected_theta: DenseVector[Double] = LogisticRegressionTest._theta
  val data: DataFrame = LogisticRegressionTest._data

  private def validateModel(model: LogisticRegressionModel): Unit = {
    model.theta.size should be(expected_theta.size)

  }

  private def validateModelAndData(model: LogisticRegressionModel, data: DataFrame): Unit = {
    validateModel(model)
    val pred = data.collect().map(_.getAs[Double](1))
    pred.length should be(LogisticRegressionTest.DATA_SIZE)

    val y_true = LogisticRegressionTest._data.collect().map(_.getAs[Double](1))
    for (i <- pred.indices) {
      pred(i) should be(y_true(i) +- delta)
    }
  }

  "Model" should "create model" in {
    val estimator = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMaxIter(100)
      .setStepSize(1.0)

    val model = estimator.fit(data)

    validateModel(model)
  }

  "Model" should "predict" in {
    val model: LogisticRegressionModel = new LogisticRegressionModel(
      theta = Vectors.fromBreeze(expected_theta).toDense,
    ).setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    validateModelAndData(model, model.transform(data))
  }


  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LogisticRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMaxIter(100)
        .setStepSize(1.0)
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
    val model = reRead.fit(data).stages(0).asInstanceOf[LogisticRegressionModel]

    validateModelAndData(model, model.transform(data))
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LogisticRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMaxIter(100)
        .setStepSize(1.0)
    ))

    val model = pipeline.fit(data)
    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModelAndData(model.stages(0).asInstanceOf[LogisticRegressionModel], reRead.transform(data))
  }
}

object LogisticRegressionTest extends WithSpark {
  val DATA_SIZE = 30000
  val VECTOR_SIZE = 3

  lazy val _theta: DenseVector[Double] = DenseVector(1.0, 1.5, 0.3, -0.7)

  lazy val _features = Seq.fill(30000)(
    Vectors.fromBreeze(DenseVector.rand(3))
  )

  lazy val generated_data: DataFrame = {
    import sqlc.implicits._
    _features.map(x => Tuple1(x)).toDF("features")
  }

  val predictionUDF = udf((features: Vector) => {
    val x = Vectors.dense(Array.concat(Array(1.0), features.toArray))
    val dotProduct = _theta dot x.asBreeze
    val prediction = sigmoid(dotProduct)
    prediction
  })


  lazy val _data =  generated_data.withColumn("label", predictionUDF(generated_data("features")))


  private def sigmoid(z: Double): Double = {
    1.0 / (1.0 + math.exp(-z))
  }
}