import java.io.{BufferedWriter, File, FileWriter}

import org.apache.commons.io.FileUtils
import org.joda.time.DateTime

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.io.Source

/**
  * Created by cxa123230 on 10/7/2017.
  */
object ChainSplitterByTimePeriod {

  def main(args: Array[String]): Unit = {
    val dir: String = "D:\\Bitcoin\\createddata\\"
    val oDir = dir + "hourly\\"
    FileUtils.cleanDirectory(new File(oDir))

    val br = Source.fromFile(dir + "txall.txt").getLines()
    val map = mutable.HashMap.empty[String, ListBuffer[String]]
    while (br.hasNext) {
      val line: String = br.next()
      val arr = line.split("\t")

      val time: DateTime = new DateTime(1000 * arr(1).toLong)
      val year = time.year().get()

      val day = time.getDayOfYear();
      val hour = time.getHourOfDay();
      val fileName: String = year + "_" + day + "_"
      if (map.contains(fileName)) {
        map(fileName).append(line + "\r\n")
        if (full(map(fileName))) {
          write(oDir + fileName + ".txt", map(fileName))
          map(fileName) = new ListBuffer()
        }
      }
      else map(fileName) = ListBuffer(line + "\r\n")

    }

    for (fileName <- map.keySet) {
      if (!map(fileName).isEmpty) {
        write(oDir + fileName + ".txt", map(fileName))
      }
    }
  }

  def full(strings: ListBuffer[String]): Boolean = {
    if (strings.length > 10000) true
    else false
  }

  def write(fileName: String, txList: ListBuffer[String]) = {
    println("writing " + fileName)
    val wr = new BufferedWriter(new FileWriter(fileName, true))
    wr.append(txList.mkString(""))
    wr.close();
  }


}
