package org.apache.spark.ml.made

import breeze.linalg.Matrix.castOps
import breeze.linalg.{sum, DenseVector => BreezeDenseVector}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap}
import org.apache.spark.ml.param.shared.{
  HasElasticNetParam, HasFeaturesCol, HasInputCol,
  HasLabelCol, HasMaxIter, HasOutputCol, HasPredictionCol, HasStepSize
}
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
    val assembledData = dataset.select(dataset($(featuresCol)), dataset($(labelCol)))
    val numFeatures: Int = MetadataUtils.getNumFeatures(assembledData, $(featuresCol))

    var theta = Vectors.dense(Array.fill(numFeatures + 1)(0.0))
    for (i <- 1 until $(maxIter)) {
      println(i)
      val grad = assembledData.rdd.map { row =>
        val label = row.getDouble(1)
        val features = row.getAs[Vector](0)
        val x = BreezeDenseVector.vertcat(new BreezeDenseVector(Array(1.0)), new BreezeDenseVector(features.toArray))
        val h = sigmoid(theta.asBreeze.dot(x))
        val error = h - label
        error * x
      }.reduce((a, b) => Vectors.dense(a.toArray.zip(b.toArray).map {
        case (x, y) => x + y
      }

      ).asBreeze.toDenseVector)

      theta = Vectors.dense(theta.toArray.zip(grad.toArray).map { case (x, y) => x - $(stepSize) * y })
    }

    copyValues(new LogisticRegressionModel(theta.toDense))
      .setParent(this)

  }

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
