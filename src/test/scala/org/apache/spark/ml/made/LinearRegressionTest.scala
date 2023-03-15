package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.made.LinearRegressionTest.{createDataFrame, sqlc}
import sqlc.implicits._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.01
  val w: DenseVector[Double] = LinearRegressionTest._w
  val b: Double = LinearRegressionTest._b
  val real: DenseVector[Double] = LinearRegressionTest._y
  val data: DataFrame = LinearRegressionTest._data

  //  System.setProperty("hadoop.home.dir","C:\\Users\\rotan\\hadoop" );
  //  var path = scala.util.Properties.envOrElse("HADOOP_HOME", "undefined")
  //  println(s"----------------------------------------------------------------- $path")
  //  System.setProperty("hadoop.home.dir", scala.util.Properties.envOrElse("HADOOP_HOME", "undefined"));
  private def validateModel(model: LinearRegressionModel): Unit = {
    model.w.size should be(w.size)
    model.w(0) should be(w(0) +- delta)
    model.w(1) should be(w(1) +- delta)
    model.w(2) should be(w(2) +- delta)

    model.b should be(b +- delta)
  }

  private def validateModelAndData(model: LinearRegressionModel, data: DataFrame): Unit = {
    validateModel(model)
    val pred = data.collect().map(_.getAs[Double](1))
    pred.length should be(LinearRegressionTest.DATA_SIZE)
    for (i <- pred.indices) {
      pred(i) should be(real(i) +- delta)
    }
  }

  "Model" should "create model" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMaxIter(100)
      .setStepSize(1.0)

    val model = estimator.fit(data)

    validateModel(model)
  }

  "Model" should "predict" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      w = Vectors.fromBreeze(w).toDense,
      b = b
    ).setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    validateModelAndData(model, model.transform(data))
  }


  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMaxIter(100)
        .setStepSize(1.0)
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)
    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    validateModelAndData(model, model.transform(data))
  }

  "Model" should "work after re-read" in {

    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
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

    validateModelAndData(model.stages(0).asInstanceOf[LinearRegressionModel], reRead.transform(data))
  }
}

object LinearRegressionTest extends WithSpark {
  val DATA_SIZE = 30000
  val VECTOR_SIZE = 3

  lazy val _data: DataFrame = createDataFrame(_X, _y)
  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand[Double](DATA_SIZE, VECTOR_SIZE)
  lazy val _w: DenseVector[Double] = DenseVector(1.5, 0.3, -0.7)
  lazy val _b: Double = 1.0
  lazy val _y: DenseVector[Double] = _X * _w + _b + DenseVector.rand(DATA_SIZE) * 0.0001

  def createDataFrame(x: DenseMatrix[Double], y: DenseVector[Double]): DataFrame = {
    lazy val data: DenseMatrix[Double] =
      DenseMatrix.horzcat(x, y.asDenseMatrix.t)

    lazy val generated_data = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq
      .toDF("x1", "x2", "x3", "label")

    lazy val assembler = new VectorAssembler().setInputCols(Array("x1", "x2", "x3")).setOutputCol("features")

    lazy val _data: DataFrame = assembler.transform(generated_data).select("features", "label")

    _data
  }
}
