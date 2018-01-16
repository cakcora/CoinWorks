import java.io.{BufferedWriter, File, FileWriter}

import org.apache.commons.io.FileUtils
import org.joda.time.DateTime
import org.slf4j.LoggerFactory

import scala.collection.mutable.ListBuffer
import scala.io.Source

/**
  * Created by cxa123230 on 10/7/2017.
  * Requires parsed transactions data from ChainParser.java
  */
object ChainSplitterByTimePeriod {
  private val log = LoggerFactory.getLogger(classOf[ChainParser])
  def main(args: Array[String]): Unit = {
    val coins = List("Bitcoin", "Litecoin", "Namecoin")
    for (coin <- coins) {
      splitData(coin)
    }
  }

  def splitData(coin: String): Unit = {
    val rootDir: String = "D:\\" + coin + "/createddata/"
    val dayF = true
    val outputDir = rootDir + {
      if (dayF) "daily/" else "hourly/"
    }
    val file: File = new File(outputDir)
    file.mkdirs()
    FileUtils.cleanDirectory(file)

    val br = Source.fromFile(rootDir + "txAll.txt").getLines()
    val content = new ListBuffer[String]()
    var previousFile = "";
    while (br.hasNext) {
      val line: String = br.next()
      val arr = line.split("\t")

      val blockDate: DateTime = new DateTime(1000 * arr(1).toLong)
      val year = blockDate.getYear()

      val day = blockDate.getDayOfYear();
      val hour = blockDate.getHourOfDay();
      val currentFile: String = if (dayF) year + "_" + day else year + "_" + day + "_" + hour
      if (previousFile.equals(currentFile)) {
        content.append(line + "\r\n")
        }

      else {
        if (previousFile.length > 1)
          write(outputDir + previousFile + ".txt", content)
        content.clear()
        content.append(line + "\r\n")
        previousFile = currentFile
      }
      }
    }



  def write(fileName: String, txList: ListBuffer[String]) = {
    new BufferedWriter(new FileWriter(fileName, true)).append(txList.mkString("")).close()
  }

}
