package copus.qa

import copus.corenlp.{CoreNlpWrapper, Document, Question}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.rdd.RDD

object TfIdf {

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\winutils")
    val sparkConf = new SparkConf().setAppName("QASystemLab2").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)

    val documents = sc.objectFile[Document]("documents.obj")

    //process each document into a sequence of lemmatized tokens
    val documentSequences = documents.map(doc => {
      doc.getTokens.map(tok => {
        tok.getLemma
      }).toSeq
    })
//    documentseq.foreach(doc => println(doc))

    val hashingTF = new HashingTF()

    val termFrequencies = hashingTF.transform(documentSequences)

    println(termFrequencies.getClass)
    termFrequencies.cache()
    termFrequencies.saveAsObjectFile("termFrequencyHashTable.obj")

//    println(hashingTF.indexOf("Mandela"))
    val idf = new IDF().fit(termFrequencies) //returns IDFModel
    val tfIdf = idf.transform(termFrequencies) //returns TF-IDF SparseVectors

    for (i : Int <- 0 until documents.count().toInt) {
      documents.collect.apply(i).setTfIdfVector(tfIdf.collect.apply(i).toSparse)
    }
    tfIdf.foreach(term => println(term))

  }//main()
}
