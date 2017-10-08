import scala.collection.mutable.ListBuffer

/**
  * Created by cxa123230 on 10/7/2017.
  */
class ChainletCluster(id: Int, s: Array[Long]) {
  val members = ListBuffer(s)
  val mergeids = ListBuffer(con(id))

  def getIds(): Set[String] = mergeids.toSet

  def add(i: Int, c: Array[Long]) {
    mergeids.append(con(i)); members.append(c)
  }

  def con(i: Int): String = {
    (1 + (i / 20)) + ":" + i % 20
    i + ""
  }

  def getMembers = {
    members.toList
  }

}
