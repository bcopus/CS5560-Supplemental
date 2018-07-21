name := "QA"
version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.1" % "provided",
  "org.apache.spark" %% "spark-streaming" % "1.6.1",
  "org.apache.spark" %% "spark-mllib" % "1.6.1",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.3.0",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.3.0" classifier "models",
  "edu.stanford.nlp" % "stanford-parser" % "3.3.0",
  "com.google.protobuf" % "protobuf-java" % "2.6.1",
  "com.github.scopt" % "scopt_2.10" % "3.4.0",
  "commons-io" % "commons-io" % "2.7-SNAPSHOT"
)