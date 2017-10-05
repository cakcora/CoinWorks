name := "CoinWorks"

version := "1.0"

scalaVersion := "2.11.9"

libraryDependencies += "joda-time" % "joda-time" % "2.9.9"
libraryDependencies  +="org.scalanlp" %% "breeze" % "0.12"
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-graphx" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.0"

resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.12-SNAPSHOT), use this.
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
