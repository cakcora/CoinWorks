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


  def getMemberVectors = {
    chainletVectors.toList
  }

}
