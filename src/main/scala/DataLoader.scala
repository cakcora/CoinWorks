import org.apache.log4j.{Level, Logger}
import org.apache.spark.graphx.{Graph, PartitionStrategy}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable
import scala.io.Source
/**
  * Created by cxa123230 on 10/2/2017.
  */
object DataLoader {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("graphboot")
      .master("local[6]")
      .getOrCreate()
    Logger.getRootLogger().setLevel(Level.ERROR)
    val sc = spark.sparkContext
    val f = loadData("C:\\Projects\\Coin\\bitcoin_dataset\\txOutAll.txt")
    println(f.size)
    val tupleGraph = Graph.fromEdgeTuples(sc.makeRDD(f), defaultValue = 1,
      uniqueEdges = Some(PartitionStrategy.RandomVertexCut))
    println(tupleGraph.numEdges)
  }
  def loadData(file:String):Array[(Long, Long)] = {
    val f = Source.fromFile(file).getLines()

    val bf = mutable.ListBuffer.empty[(Long,Long)]
     for(line<-f){
      val arr = line.split("\t")
     for(s<-arr(1).split(","))
      bf.append((arr(0).toLong, s.toLong))
    }
    bf.toArray
  }

}
