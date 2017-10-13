
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics

import scala.collection.mutable.ListBuffer

/**
  * Created by cxa123230 on 10/7/2017.
  */
class ChainletCluster(thisId: Int, thisVector: Array[Long]) {
  val chainletVectors = ListBuffer(thisVector)
  val mergedChainletIds = ListBuffer(thisId)

  def addId(chainletId: Int) = {
    mergedChainletIds.append(chainletId);
  }

  def getMemberIds(): Set[Int] = mergedChainletIds.toSet

  def add(vector: Array[Long]) {
    chainletVectors.append(vector)
  }

  def getClusterVector(method: String): Array[Double] = {
    val vec = Array.ofDim[Double](chainletVectors(0).length)
    for (i <- 0 to chainletVectors(0).length - 1) {
      val x = new DescriptiveStatistics()
      for (v <- chainletVectors) {
        x.addValue(v(i))
      }

      if (method == "avg") {
        vec(i) = x.getMean
      }
      if (method == "max") {
        vec(i) = x.getMax
      }
      if (method == "median") {
        vec(i) = x.getPercentile(0.5)
      }
    }
    vec
  }

  def getMemberVectors = {
    chainletVectors.toList
  }

}
