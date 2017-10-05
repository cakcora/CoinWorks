import java.io.FileWriter

import org.joda.time.DateTime

import scala.collection.mutable.HashMap
import scala.collection.mutable.ListBuffer
import scala.io.Source

/**
  * Created by cxa123230 on 9/29/2017.
  */
object GraphParser {

  def main(args: Array[String]): Unit = {
    val timeFile = "C:\\Projects\\Coin\\bitcoin_dataset\\txtime.txt"
    val inFile = "C:\\Projects\\Coin\\bitcoin_dataset\\txin.txt"
    val outFile = "C:\\Projects\\Coin\\bitcoin_dataset\\txout.txt"
    val MONTH = true
    val DAY = !MONTH
     val times:Map[Long, Long] =  Source.fromFile(timeFile).getLines().map(e=>e.split("\t")).map(e=>e(0).toLong->e(1).toLong).toMap
    println(times.size +" "+times(30048982L))

    val minCoinFlow: Long = 0L
    joinData(inFile,"txInAll.txt",times,minCoinFlow)
    joinData(outFile,"txOutAll.txt",times,minCoinFlow)

  }

  def joinData(inFile:String, outputFile:String, times:Map[Long, Long],minCoin:Long): Unit ={
    var exTxId = -1L;
    val bf = ListBuffer.empty[Long]
    var sum =0L;
    var i =0
    val inMap:HashMap[Long, ListBuffer[Long]] = HashMap.empty[Long,ListBuffer[Long]]
    val fw = new FileWriter("C:\\Projects\\Coin\\bitcoin_dataset\\"+outputFile)
    for(line <- Source.fromFile(inFile).getLines()){
      //txid, add, value
      val arr = line.split("\t")
      val txId:Long = arr(0).toLong
      val address: Long = arr(1).toLong
      val amount: Long = arr(2).toLong

      if(txId!=exTxId){
        if(exTxId!= -1L){
          fw.append(exTxId+"\t"+times(exTxId)+"\t"+sum+"\t"+bf.mkString(",")+"\r\n")
        }
        exTxId = txId
        sum =amount;
        bf.clear()
        bf.append(address)
        i +=1
        if(i%100000==0)
          println(i)

      }
      else {

        if (amount > minCoin) {
          sum += amount
          bf.append(address)
        }
      }

    }
    fw.append(exTxId+"\t"+times(exTxId)+"\t"+bf.mkString(",")+"\r\n")
    fw.close()

  }
}
