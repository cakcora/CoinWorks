name := "CoinWorks"

version := "1.0"

scalaVersion := "2.11.9"

libraryDependencies += "joda-time" % "joda-time" % "2.9.9"
libraryDependencies  +="org.scalanlp" %% "breeze" % "0.12"
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-graphx" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.0"
libraryDependencies += "com.google.code.gson" % "gson" % "1.7.1"
libraryDependencies += "org.bitcoinj" % "bitcoinj-core" % "0.14.5"
libraryDependencies += "org.bitcoinj" % "bitcoinj-tools" % "0.14.5"

libraryDependencies += "com.github.zuinnote" % "hadoopcryptoledger-fileformat" % "1.1.2" % "compile"
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-graphx" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.0"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test,it"
libraryDependencies += "javax.servlet" % "javax.servlet-api" % "3.0.1" % "it"
libraryDependencies += "org.apache.hadoop" % "hadoop-common" % "2.7.0" % "it" classifier "" classifier "tests"
libraryDependencies += "org.apache.hadoop" % "hadoop-hdfs" % "2.7.0" % "it" classifier "" classifier "tests"
libraryDependencies += "org.apache.hadoop" % "hadoop-minicluster" % "2.7.0" % "it"


resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.12-SNAPSHOT), use this.
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
