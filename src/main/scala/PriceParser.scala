import java.io.IOException
import java.net.SocketTimeoutException

import scala.io.Source
import org.joda.time.DateTime


/**
  * Created by cxa123230 on 9/28/2017.
  */
object PriceParser {

  def main(args: Array[String]): Unit = {
  getPrice()
  val blockId = "0000000000000bae09a7a393a8acded75aa67e46cb81f7acaa5ad94f9eacd103"
    try {
      val content = get("https://blockchain.info/rawblock/"+blockId)
      println(content)
    } catch {
      case ioe: IOException =>  // handle this
      case ste: SocketTimeoutException => // handle this
    }
  }
  def getPrice(): Unit = {
    val filename = "C:\\Projects\\Coin\\bitcoin-historical-data\\bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv"
    var change = -1.0
    var current = 0
    var N = 0
    val MONTH = true
    val DAY = !MONTH

    for (line <- Source.fromFile(filename).getLines().drop(1)) {
      val arr = line.split(",")
      val dateTime: DateTime = new DateTime(arr(0).toLong * 1000).toDateTime
      val resolution: Int = if (MONTH) dateTime.monthOfYear().get() else if (DAY) dateTime.dayOfYear().get() else -1
      val year = dateTime.year().get()
      val close = arr(4).toDouble
      if (!close.isNaN || arr.length != 8) {
        if (resolution != current) {
          println(year + " " + current + " " + change / N + " " + N)
          current = resolution
          N = 1
          change = close

        }
        else {
          N += 1
          change += close
        }
      }
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
