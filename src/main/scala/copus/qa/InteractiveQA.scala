package copus.qa

import scala.io.StdIn.readLine
import copus.corenlp.{CoreNlpWrapper, Question, Sentence}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.{SparkConf, SparkContext}

object InteractiveQA {

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\winutils")
    val sparkConf = new SparkConf().setAppName("QASystemLab3").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)

    val hashingTF = new HashingTF()
    val sentences = sc.objectFile[Sentence]("preparedCorpus\\sentences.obj")
    CoreNlpWrapper.initializePipeline()

    var exit = false;
    do {
      val qtext : String = readLine("Question: ")
      val question = new Question(qtext, hashingTF)
      val bestSentences = sentences.map(sentence => Tuple2(sentence, sentence.score(question)))
        .filter(tup => tup._2 > 0.0)
        .sortBy(tup => tup._2, ascending = false)
        .take(1)
      val candidateAnswers = bestSentences.map(sentence => sentence._1.extractAnswer(question))
      candidateAnswers.foreach(answer => println("\nAnswer: " + answer))
    } while (!exit)
  }//main()

}//InteractiveQA
