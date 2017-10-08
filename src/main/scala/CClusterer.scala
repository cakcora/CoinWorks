import java.io.File

import scala.collection.mutable
import scala.io.Source

/**
  * Created by cxa123230 on 10/7/2017.
  */
object CClusterer {
  val timePeriodMax: Int = 365

  def main(args: Array[String]): Unit = {
    val crdir: String = "D:\\Bitcoin\\createddata\\dailyOccMatrices\\"
    val startYear: Int = 2009
    val records = Array.ofDim[Long](400, (2017 - startYear + 1) * 365)
    var currentTimePeriod = 0
    for (year: Int <- startYear to 2017 by 1) {
      for (timePeriod <- 1 to timePeriodMax by 1) {
        val fileName: String = crdir + "occ" + year + "day" + timePeriod + ".csv"
        val f: File = new File(fileName)

        if (f.exists) {
          val lines = Source.fromFile(fileName).getLines().toList
          var currentChainlet = 0;
          for (line <- lines) {
            val arr = line.split(",")
            for (c <- arr) {
              records(currentChainlet)(currentTimePeriod) = c.toLong
              //              println(currentChainlet+" "+currentTimePeriod+" "+c.toInt)
              currentChainlet += 1
            }
          }
        }
        else {
          println(fileName + " :((")
        }
        currentTimePeriod += 1
      }
    }


    //    var maxSim =0.0
    //    var cand = -1
    //    for(r<-0 to records.length-1){
    //        for(r2<-r+1 to records.length-1){
    //          val similarity = CosineSim.cosineSimilarity(records(r),records(r2))
    //          if(similarity>maxSim){
    //            maxSim=similarity
    //            cand = -1
    //          }
    //        }
    //    }
    val clusters = mutable.HashMap.empty[Int, ChainletCluster]
    for (r <- 0 to records.length - 1) {
      clusters(r) = new ChainletCluster(r, records(r))
    }
    var s = 0
    val sims = Array.ofDim[Double](400, 400)
    val bf = new StringBuffer()
    while (clusters.size != 1) {
      s += 1
      println("step " + s)
      var maxSim = -1.0
      var merge1 = -1
      var merge2 = -1
      for (cc1 <- clusters.keySet) {
        for (cc2 <- clusters.keySet) {
          val c1 = clusters(cc1)
          val c2 = clusters(cc2)
          if (c1 != c2) {
            var avgSim = 0.0
            var i = 0
            if (sims(cc1)(cc2) == 0 || sims(cc1)(cc2) == -1) {
              //              println("computing "+cc1+" "+cc2)
              for (m1 <- c1.getMembers) {
                for (m2 <- c2.getMembers) {
                  avgSim += CosineSim.cosineSimilarity(m1, m2)
                  i += 1
                }
              }
              avgSim = avgSim / i
              sims(cc1)(cc2) = avgSim
            }
            avgSim = sims(cc1)(cc2)
            if (avgSim > maxSim) {
              maxSim = avgSim
              merge1 = cc1
              merge2 = cc2
              println(cc1 + " " + cc2 + " " + maxSim + " maxSim")
            }
          }
        }
      }

      if (maxSim < 0.7) {

        for (key <- clusters.keySet) {
          println(clusters(key).getIds().mkString(","))
        }
        println(clusters.size + " clusters")
        System.exit(1)
      }
      val cluster1 = clusters(merge1)
      val cluster2 = clusters(merge2)
      for (k <- clusters.keySet) {
        sims(merge1)(k) = -1

      }
      for (member <- cluster2.getMembers) {
        cluster1.add(merge2, member)
      }
      clusters(merge1) = cluster1
      println("(" + merge1 + "," + merge2 + "):" + maxSim + " remaining " + clusters.size)
      clusters.remove(merge2)

    }


  }

}
