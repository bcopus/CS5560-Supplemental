package copus.qa

import copus.corenlp.{Document, Question}
import org.apache.spark.{SparkConf, SparkContext}

object DocViewer {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\winutils")
    val sparkConf = new SparkConf().setAppName("QASystemLab2").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)

    val documents = sc.objectFile[Document]("documents.obj")
    documents.foreach(doc => println(doc))
  }//main

}//DocViewer
