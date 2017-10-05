import org.apache.spark.sql.SparkSession
import org.apache.log4j.{Level, Logger}
/**
  * Created by cxa123230 on 10/2/2017.
  */
object GraphletMiner {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("graphboot")
      .master("local[6]")
      .getOrCreate()
    Logger.getRootLogger().setLevel(Level.ERROR)
    val sc = spark.sparkContext
  }
}
