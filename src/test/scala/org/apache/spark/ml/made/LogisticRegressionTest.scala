//package org.apache.spark.ml.made
//
//import breeze.linalg.{*, DenseMatrix, DenseVector}
////import breeze.linalg.{sum, DenseMatrix => BreezeDenseMatrix, DenseVector => BreezeDenseVector}
//import breeze.optimize.DiffFunction.castOps
//import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
//import com.google.common.io.Files
//import org.apache.spark.ml.feature.VectorAssembler
//import org.apache.spark.ml.linalg.Vectors
//import org.apache.spark.ml.made.LinearRegressionTest.sqlc
//import org.apache.spark.ml.util.Identifiable
//import org.apache.spark.ml.{Pipeline, PipelineModel}
//import org.apache.spark.sql.DataFrame
//import org.apache.spark.sql.functions.udf
//import org.scalatest._
//import org.scalatest.flatspec._
//import org.scalatest.matchers._


package org.apache.spark.ml.made

//import org.apache.spark.implicits._
//import sqlc.implicits._

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
  //  val b: Double = LinearRegressionTest._b
  //  val expected_labels: DenseVector[Double] = LogisticRegressionTest._labels
  val data: DataFrame = LogisticRegressionTest._data

  //  System.setProperty("hadoop.home.dir","C:\\Users\\rotan\\hadoop" );
  //  var path = scala.util.Properties.envOrElse("HADOOP_HOME", "undefined")
  //  println(s"----------------------------------------------------------------- $path")
  //  System.setProperty("hadoop.home.dir", scala.util.Properties.envOrElse("HADOOP_HOME", "undefined"));
  private def validateModel(model: LogisticRegressionModel): Unit = {
    model.theta.size should be(expected_theta.size)
//    model.theta(0) should be(expected_theta(0) +- delta)
//    model.theta(1) should be(expected_theta(1) +- delta)
//    model.theta(2) should be(expected_theta(2) +- delta)

    //    model.b should be(b +- delta)
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

  //  lazy val _data: DataFrame = createDataFrame(_X, _labels)
  //  lazy val _data: DataFrame = transform_data(_X)
  //  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand[Double](DATA_SIZE, VECTOR_SIZE)
  lazy val _theta: DenseVector[Double] = DenseVector(1.0, 1.5, 0.3, -0.7)
  //  val _labels = _theta.dot(_X)
  //  val predictUdf = _data.sqlContext.udf.register(Identifiable.randomUID("_transform"),
  //    (x: Vector) => {
  //      (_theta.asBreeze dot x.asBreeze)
  //    })
  lazy val _features = Seq.fill(30000)(
    Vectors.fromBreeze(DenseVector.rand(3))
  )

  lazy val generated_data: DataFrame = {
    import sqlc.implicits._
    _features.map(x => Tuple1(x)).toDF("features")
  }

  //  def transform_data(_x: DenseMatrix[Double]): DataFrame = {
  //    lazy val generated_data = _x(*, ::).iterator
  //      .map(x => (x(0), x(1), x(2)))
  //      .toSeq
  //      .toDF("x1", "x2", "x3")

  //    val assembler = new VectorAssembler()
  //      .setInputCols(Array("x1", "x2", "x3"))
  //      .setOutputCol("features")

  //    generated_data.show(false)
  //    println(generated_data("x1"))
  //    assembledData.show(false)
  //
  //    generated_data.show(false)
  //    val assembledData = assembler.transform(generated_data)
  //      .select("features")
  //      .rdd
  //      .map(row => row.getAs[Vector]("features"))

  val predictionUDF = udf((features: Vector) => {
    val x = Vectors.dense(Array.concat(Array(1.0), features.toArray))
    val dotProduct = _theta dot x.asBreeze
    val prediction = sigmoid(dotProduct)
    prediction
  })

  import sqlc.implicits._

  // Сырые данные
  //    val assembledData = assembler.transform(generated_data).select("features")
//  lazy val tmp_data = _features.map(x => (x(0), x(1), x(2))).toDF("x1", "x2", "x3")
//  lazy val gg = generated_data.withColumn("label", predictionUDF(generated_data("features"))).drop("features")
//  val _data = tmp_data.join(gg)

  lazy val _data =  generated_data.withColumn("label", predictionUDF(generated_data("features")))

//  _data.show()

  //
////  val merged_df = tmp_data.unionByName(gg("label"))
//  tmp_data.show(false)
//  import org.apache.spark.sql.functions._
////  println(gg.select("label").map(r => r(0).asInstanceOf[Double]).collect())
//  lazy val _data = tmp_data.withColumn("label", gg("label"))
////  _data.show(false)
////  lazy val _data =
//  println(8)
  //    println(assembledData.select("features")("features"))
  //    val schemaSeq = Seq("empno", "ename", "designation", "manager")
  //
  //    //Create Empty DataFrame using Seq
  //    val emptyDF3 = Seq.empty[(String, String, String, String)].toDF(schemaSeq: _*)
  //    assembledData.select()
  //    val tt = predictionUDF(assembledData("features"))
  //    println(tt)
  //    assembledData.explode(assembledData("features"))
  //    emptyDF3.withColumn("labels", predictionUDF(assembledData("features")))
  //    generated_data.withColumn("labels", predictionUDF(assembledData("features"))).show(false)

  //  lazy val _data =  generated_data.withColumn("label", predictionUDF(generated_data("features")))

  //    generated_data.withColumn("labels", assembledData
  //      .map(row => predictionUDF(row("features"))).toDF() )
  ////    generated_data.show(false)
  //    generated_data
  //  }

  private def sigmoid(z: Double): Double = {
    1.0 / (1.0 + math.exp(-z))
  }
}
//    assembler.transform(generated_data)
//      .select("prediction")
//      .rdd
//      .map(row => row.getAs[DenseVector]("prediction"))
//
//  def createDataFrame(x: DenseMatrix[Double], y: DenseVector[Double]): DataFrame = {
//    lazy val data: DenseMatrix[Double] =
//      DenseMatrix.horzcat(x, y.asDenseMatrix.t)
//
//    lazy val generated_data = data(*, ::).iterator
//      .map(x => (x(0), x(1), x(2), x(3)))
//      .toSeq
//      .toDF("x1", "x2", "x3", "label")
//
//    lazy val assembler = new VectorAssembler().setInputCols(Array("x1", "x2", "x3")).setOutputCol("features")
//
//    lazy val _data: DataFrame = assembler.transform(generated_data).select("features", "label")
//
//    _data
//  }
//

//  val sigmoidUdf = udf((z: Double) => 1.0 / (1.0 + math.exp(-z)))
//  lazy val _features = Seq.fill(100000)(
//    BreezeDenseVector.rand(3)
//  )
//  //  lazy val _features = DenseMatrix.rand[Double](DATA_SIZE, VECTOR_SIZE)
//  //  lazy val _theta =  Vectors.dense(1.5, 0.3, -0.7).asBreeze
//  lazy val _theta = BreezeDenseVector(1.5, 0.3, -0.7)
//  //  lazy val _labels =  sigmoidUdf((_theta.dot(_features)))
//  //  lazy val _data: DataFrame = createDataFrame(_features, _labels)
//  //  lazy val _b: Double = 1.0 bnm
//
//  lazy val _data: DataFrame = {
//    import sqlc.implicits._
//    _features.map(x => Tuple1(x)).toDF("features")
//  }
//
//  lazy val transformUdf =
//    udf((x: Vector) => {
//      _theta dot x.asBreeze
//    })
//
//  //  val data = _data.withColumn("features", transformUdf(_data("features")))
//
//
//  val predictUdf = _data.sqlContext.udf.register(Identifiable.randomUID("_transform"),
//    (x: Vector) => {
//      (_theta dot x.asBreeze)
//    })
//
//  lazy val _labels = predictUdf(_data("label"))

//  def createDataFrame(x: DenseMatrix[Double], y: DenseVector[Double]): DataFrame = {
//    import sqlc.implicits._
//    lazy val data: DenseMatrix[Double] =
//      DenseMatrix.horzcat(x, y.asDenseMatrix.t)
//
//    lazy val generated_data = data(*, ::).iterator
//      .map(x => (x(0), x(1), x(2), x(3)))
//      .toSeq
//      .toDF("x1", "x2", "x3", "label")
//
//    lazy val assembler = new VectorAssembler().setInputCols(Array("x1", "x2", "x3")).setOutputCol("features")
//
//    lazy val transformUdf =
//      udf((x: Vector) => {
//        _theta dot x.asBreeze
//      })
//    lazy val _data = _data.withColumn("target", transformUdf(_data("features")))
//    lazy val _data: DataFrame = assembler.transform(generated_data).select("features", "label")
//
//    _data
//  }


//
//object LinearRegressionTest extends WithSpark {
//  val DATA_SIZE = 30000
//  val VECTOR_SIZE = 3
//  val sigmoidUdf = udf((z: Double) => 1.0 / (1.0 + math.exp(-z)))
//  lazy val _features = Seq.fill(100000)(
//    BreezeDenseVector.rand(3)
//  )
//  //  lazy val _features = DenseMatrix.rand[Double](DATA_SIZE, VECTOR_SIZE)
//  //  lazy val _theta =  Vectors.dense(1.5, 0.3, -0.7).asBreeze
//  lazy val _theta = BreezeDenseVector(1.5, 0.3, -0.7)
//  //  lazy val _labels =  sigmoidUdf((_theta.dot(_features)))
//  //  lazy val _data: DataFrame = createDataFrame(_features, _labels)
//  //  lazy val _b: Double = 1.0 bnm
//
//  lazy val _data: DataFrame = {
//    import sqlc.implicits._
//    _features.map(x => Tuple1(x)).toDF("features")
//  }
//
//  lazy val transformUdf =
//    udf((x: Vector) => {
//      _theta dot x.asBreeze
//    })
//
////  val data = _data.withColumn("features", transformUdf(_data("features")))
//
//
//  val predictUdf = _data.sqlContext.udf.register(Identifiable.randomUID("_transform"),
//    (x: Vector) => {
//      (_theta dot x.asBreeze)
//    })
//
//  lazy val _labels = predictUdf(_data("label"))
//
//  //  def createDataFrame(x: DenseMatrix[Double], y: DenseVector[Double]): DataFrame = {
//  //    import sqlc.implicits._
//  //    lazy val data: DenseMatrix[Double] =
//  //      DenseMatrix.horzcat(x, y.asDenseMatrix.t)
//  //
//  //    lazy val generated_data = data(*, ::).iterator
//  //      .map(x => (x(0), x(1), x(2), x(3)))
//  //      .toSeq
//  //      .toDF("x1", "x2", "x3", "label")
//  //
//  //    lazy val assembler = new VectorAssembler().setInputCols(Array("x1", "x2", "x3")).setOutputCol("features")
//  //
//  //    lazy val transformUdf =
//  //      udf((x: Vector) => {
//  //        _theta dot x.asBreeze
//  //      })
//  //    lazy val _data = _data.withColumn("target", transformUdf(_data("features")))
//  //    lazy val _data: DataFrame = assembler.transform(generated_data).select("features", "label")
//  //
//  //    _data
//  //  }
//}
