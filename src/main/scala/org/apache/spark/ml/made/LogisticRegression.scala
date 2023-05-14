package org.apache.spark.ml.made

import breeze.linalg.Matrix.castOps
import breeze.linalg.{sum, DenseVector => BreezeDenseVector}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{HasElasticNetParam, HasFeaturesCol, HasInputCol, HasLabelCol, HasMaxIter, HasOutputCol, HasPredictionCol, HasStepSize}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.udf
import org.apache.spark.mllib
import org.apache.spark.mllib.linalg.Vectors.fromBreeze
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame

import javax.crypto.interfaces.DHPublicKey

trait LogisticRegressionParams extends HasLabelCol
  with HasFeaturesCol
  with HasPredictionCol
  with HasMaxIter
  with HasStepSize
  with HasElasticNetParam {

  def setLabelCol(value: String): this.type = set(labelCol, value)

  //  def setElasticNetParam(value: Int): this.type = set(elasticNetParam, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)


  //  setDefault(maxIter -> 500, stepSize -> 0.01, elasticNetParam -> 0.8)
  setDefault(maxIter -> 500, stepSize -> 0.01)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, DoubleType)
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
    }
  }
}

class LogisticRegression(override val uid: String) extends Estimator[LogisticRegressionModel]
  with LogisticRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("logisticRegression"))

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setStepSize(value: Double): this.type = set(stepSize, value)


  override def fit(dataset: Dataset[_]): LogisticRegressionModel = {
    //    val assembler = new VectorAssembler()
    //      .setInputCols(Array("feature1", "feature2", "feature3"))
    //      .setOutputCol("features")

    //    val assembledData = assembler.transform(dataset).select("features", "label")
    //    dataset.show(false)
    //    val assembler: VectorAssembler = new VectorAssembler()
    //      .setInputCols(Array($(featuresCol), "ones", $(labelCol)))
    //      .setOutputCol("features_ext")
    //
    ////    val datasetExt: Dataset[_] = dataset.withColumn("ones", lit(1))
    //    val assembledData = assembler.transform(datasetExt).select("features_ext")
    //    assembledData.show(false)
//    val features = dataset.columns.filterNot(_ == $(labelCol))
//    val assembler = new VectorAssembler()
//      .setInputCols(features)
//      .setOutputCol("features")
//
//
//    val assembledData = assembler.transform(dataset).select("features", "label")

    val assembledData = dataset.select(dataset($(featuresCol)), dataset($(labelCol)))
    val numFeatures: Int = MetadataUtils.getNumFeatures(assembledData, $(featuresCol))
    //    val m = assembledData.first().getAs[Vector](0).size


    var theta = Vectors.dense(Array.fill(numFeatures + 1)(0.0))
    //    var theta: BreezeDenseVector[Double] = BreezeDenseVector.rand[Double](numFeatures + 1)
    for (i <- 1 until $(maxIter)) {
      println(i)
      val grad = assembledData.rdd.map { row =>
        val label = row.getDouble(1)
        val features = row.getAs[Vector](0)
        val x = BreezeDenseVector.vertcat(new BreezeDenseVector(Array(1.0)), new BreezeDenseVector(features.toArray))
//        val x = Vectors.dense(Array.concat(Array(1.0), features.toArray))
        val h = sigmoid(theta.asBreeze.dot(x))
        val error = h - label
//        val h = sigmoid(theta.asBreeze.dot(x))
        //        error * x.toDenseVector
        error * x
      }.reduce((a, b) => Vectors.dense(a.toArray.zip(b.toArray).map {
        case (x, y) => x + y
      }

      ).asBreeze.toDenseVector)

      theta = Vectors.dense(theta.toArray.zip(grad.toArray).map { case (x, y) => x - $(stepSize) * y })
    }


    //    var theta: BreezeDenseVector[Double] = BreezeDenseVector.rand[Double](numFeatures + 1)
    //    for (i <- 1 to $(maxIter)) {
    //      val grad = assembledData.rdd.map { row =>
    //          val label = row.getDouble(1)
    //          val features = row.getAs[Vector](0)
    //          val x = BreezeDenseVector.vertcat(new BreezeDenseVector(Array(1.0)), new BreezeDenseVector(features.toArray))
    //          val h = sigmoid(theta.dot(x)) cv
    //          val error = h - label
    //          error * x
    //      }.reduce(_ + _)
    //
    //      theta -= $(stepSize) * grad
    //    }

    //
    //    for (i <- 1 to $(maxIter)) {
    //      val grad = vectors.map { row =>
    //        val label = row.getDouble(1)
    //        val features = row.getAs[Vector](0)
    //        //        val x = DenseVector.vertcat(new DenseVector(Array(1.0)), new DenseVector(features.toArray))
    //        val x = Vectors.dense(Array.concat(Array(1.0), features.toArray))
    //        val h = sigmoid(theta.dot(x))
    //        val error = h - label
    //        error * x
    //      }.reduce((a, b) => Vectors.dense(a.toArray.zip(b.toArray).map { case (x, y) => x + y }))
    //
    //      theta -= $(stepSize) * grad
    //    }
    //    copyValues(new LogisticRegressionModel(Vectors.fromBreeze(theta(0 until theta.size - 1)).toDense))
    //      .setParent(this)

    copyValues(new LogisticRegressionModel(theta.toDense))
      .setParent(this)
    //
    //    copyValues(new LogisticRegressionModel(Vectors.fromBreeze(theta(0 until theta.size - 1)).toDense))
    //          .setParent(this)
  }


  //  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
  //
  //    // Used to convert untyped dataframes to datasets with vectors
  ////    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
  //    val datasetExt: Dataset[_] = dataset.withColumn("ones", lit(1))
  //    val numFeatures: Int = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
  //
  //    val assembler: VectorAssembler = new VectorAssembler()
  //      .setInputCols(Array($(featuresCol), "ones", $(labelCol)))
  //      .setOutputCol("features_ext")
  //
  ////    val vectors: Dataset[Vector] = assembler.transform(datasetExt).select("features_ext").as[Vector]
  //    val vectors = assembler.transform(datasetExt).select("features_ext")
  //    var theta: BreezeDenseVector[Double] = BreezeDenseVector.rand[Double](numFeatures + 1)
  //
  //
  //    for (i <- 1 to $(maxIter)) {
  //      val grad = vectors.map { row =>
  //        val label = row.getDouble(1)
  //        val features = row.getAs[Vector](0)
  ////        val x = DenseVector.vertcat(new DenseVector(Array(1.0)), new DenseVector(features.toArray))
  //        val x = Vectors.dense(Array.concat(Array(1.0), features.toArray))
  //        val h = sigmoid(theta.dot(x))
  //        val error = h - label
  //        error * x
  //      }.reduce((a, b) => Vectors.dense(a.toArray.zip(b.toArray).map { case (x, y) => x + y }))
  //
  //      theta -= $(stepSize) * grad
  //    }
  //    copyValues(new LinearRegressionModel(Vectors.fromBreeze(theta(0 until theta.size - 1)).toDense))
  //      .setParent(this)
  //
  ////
  //    for (i <- 1 to $(maxIter)) {
  //      val gradient = dataset.rdd.map(point => (point.label - predict(point.features, weights)) * point.features)
  //        .reduce((a, b) => Vectors.asBreeze((0 until numFeatures).map(i => a(i) + b(i)).toArray))
  //      weights = Vectors.dense((0 until numFeatures).map(i => weights(i) + stepSize * gradient(i)).toArray)
  //    }
  ////
  ////    weights
  ////
  ////
  ////    for (_ <- 0 until $(maxIter)) {
  ////      val summary = vectors.rdd
  ////        .mapPartitions((data: Iterator[Vector]) => {
  ////          val summarizer = new MultivariateOnlineSummarizer()
  ////          data.foreach(vector => {
  ////            val X = vector.asBreeze(0 until w.size).toDenseVector
  ////            val y = vector.asBreeze(w.size)
  ////            summarizer.add(fromBreeze(X * (sum(X * w) - y)))
  ////          })
  ////          Iterator(summarizer)
  ////        })
  ////        .reduce(_ merge _)
  ////      w = w - $(stepSize) * summary.mean.asBreeze
  ////    }
  //
  ////    copyValues(new LinearRegressionModel(Vectors.fromBreeze(w(0 until w.size - 1)).toDense,w(w.size - 1)))
  ////      .setParent(this)
  //  }

  private def sigmoid(z: Double): Double = {
    1.0 / (1.0 + math.exp(-z))
  }

  override def copy(extra: ParamMap): Estimator[LogisticRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LogisticRegression extends DefaultParamsReadable[LogisticRegression]

class LogisticRegressionModel private[made](
                                             override val uid: String,
                                             val theta: DenseVector)
  extends Model[LogisticRegressionModel]
    with LogisticRegressionParams
    with MLWritable {


  private[made] def this(theta: DenseVector) =
    this(Identifiable.randomUID("logisticRegressionModel"), theta.toDense)

  override def copy(extra: ParamMap): LogisticRegressionModel = copyValues(
    new LogisticRegressionModel(theta), extra)

  def sigmoid(z: Double): Double = {
    1.0 / (1.0 + math.exp(-z))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val assembler = new VectorAssembler()
      .setInputCols(Array("features"))
      .setOutputCol("featuresVector")

    val assembledData = assembler.transform(dataset)
      .select("featuresVector")
      .rdd
      .map(row => row.getAs[Vector]("featuresVector"))

    val predictionUDF = udf((features: Vector) => {
      val x = Vectors.dense(Array.concat(Array(1.0), features.toArray))
      val dotProduct = theta.dot(x)
      val prediction = sigmoid(dotProduct)
      prediction
    })

    dataset.withColumn($(predictionCol), predictionUDF(dataset($(featuresCol))))
  }

  //  override def transform(dataset: Dataset[_]): DataFrame = {
  //    val sigmoidUdf = udf((z: Double) => 1.0 / (1.0 + math.exp(-z)))
  //
  ////    val predictUdf = udf((features: Vector) => {
  ////        val x = Vectors.dense(Array.concat(Array(1.0), features.toArray))
  ////        sigmoidUdf(theta.dot(x))
  ////    })
  //
  //    val predictUdf = dataset.sqlContext.udf.register(uid + "_transform",
  //      (x: Vector) => {
  //        sigmoidUdf((theta.asBreeze dot x.asBreeze).toDouble))
  //      })
  //
  ////    dataset.withColumn("prediction", predictUdf($"features"))
  //    dataset.withColumn($(predictionCol), predictUdf(dataset($(featuresCol))))

  //    val transformUdf =
  //      dataset.sqlContext.udf.register(uid + "_transform",
  //        (x: Vector) => {
  //          Vectors.fromBreeze(BreezeDenseVector(w.asBreeze.dot(x.asBreeze) + b))
  //        }
  //      )
  //
  //    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  //  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors: (Vector, Int) = theta.asInstanceOf[Vector] -> 1

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}

object LogisticRegressionModel extends MLReadable[LogisticRegressionModel] {
  override def read: MLReader[LogisticRegressionModel] = new MLReader[LogisticRegressionModel] {
    override def load(path: String): LogisticRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val theta = vectors.select(vectors("_1").as[Vector]).first()

      val model = new LogisticRegressionModel(theta.toDense)
      metadata.getAndSetParams(model)
      model
    }
  }
}
