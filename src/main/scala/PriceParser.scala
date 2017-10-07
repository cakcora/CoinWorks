import java.io.{IOException, PrintWriter}
import java.net.SocketTimeoutException

import com.google.gson.JsonParser

import scala.io.Source


/**
  * Created by cxa123230 on 9/28/2017.
  */

object PriceParser {

  def main(args: Array[String]): Unit = {
    val ldir = "D:\\Litecoin\\BlocksFromApi\\"
    val bdir = "D:\\Bitcoin\\BlocksFromApi\\"

    var sleeptime = 200


    if(false)
    startBitCoinDownload("0000000000000000001eb6d2dc29a3d0cca42cb704ad5815cec73a83b1d694d9",bdir,sleeptime)

    startLiteCoinDownload(1290340,ldir,sleeptime)

  }
  def startLiteCoinDownload(b:Long, dir:String, s:Long): Unit = {
    var i =0;
    var sleeptime = s
    var blockNo = b
    var prev_block = -1L
    for (x <- 1 to (2017 - 2014) * 365 * 24 * 6) {
      prev_block = getLiteCoinBlock(dir, blockNo)
      if (prev_block != -1) {
        i += 1
        println("LTC "+i + " " + blockNo + " " + prev_block)
        blockNo = -1
        sleeptime = 800
      }
      else {
        sleeptime = 2 * sleeptime
        println("i will wait (s) " + sleeptime / 1000)
      }
      Thread.sleep(sleeptime)
    }
  }
  def startBitCoinDownload(block:String,dir:String,sleep:Long): Unit = {
    var i =0;
    var sleeptime = sleep
    var blockHash = block
    var prev_blockhash = ""
    for (x <- 1 to (2017 - 2014) * 365 * 24 * 6) {
      prev_blockhash = getBitCoinBlock(dir, blockHash)
      if (prev_blockhash != "null") {
        i += 1
        println("BTC "+i + " " + blockHash + " " + prev_blockhash)
        blockHash = prev_blockhash
        sleeptime = 800
      }
      else {
        sleeptime = 2 * sleeptime
        println("i will wait (s) " + sleeptime / 1000)
      }
      Thread.sleep(sleeptime)
    }
  }
  def getLiteCoinBlock(dir: String, blockNo: Long): Long = {
    try {
      val add = "http://chain.so/api/v2/block/LTC/"
      println(add+ blockNo)
      val content = get( add+ blockNo)

      new PrintWriter(dir + blockNo + ".json") {
        write(content)
        close()
      }
      val prev_block = blockNo-1L
      prev_block
    } catch {
      case e:Exception =>{println(e.toString); -1}
      case ioe: IOException => -1
      case ste: SocketTimeoutException => -1
    }

  }

  def getBitCoinBlock(dir: String, blockHash: String): String = {
    try {
      val add = "https://blockchain.info/rawblock/"
      val content = get( add+ blockHash)
      //

      val parser = new JsonParser()
      val o = parser.parse(content).getAsJsonObject()
      new PrintWriter(dir + blockHash + ".json") {
        write(content)
        close()
      }
      val prev_block = o.get("prev_block").getAsString
      prev_block
    } catch {
      case ioe: IOException => "null"
      case ste: SocketTimeoutException => "null"
    }

  }

  @throws(classOf[java.io.IOException])
  @throws(classOf[java.net.SocketTimeoutException])
  def get(url: String,
          connectTimeout: Int = 5000,
          readTimeout: Int = 5000,
          requestMethod: String = "GET") =
  {
    import java.net.{URL, HttpURLConnection}
    val connection = (new URL(url)).openConnection.asInstanceOf[HttpURLConnection]
    connection.setConnectTimeout(connectTimeout)
    connection.setReadTimeout(readTimeout)
    connection.setRequestMethod(requestMethod)
    val inputStream = connection.getInputStream
    val content = Source.fromInputStream(inputStream).mkString
    if (inputStream != null) inputStream.close
    content
  }
}
