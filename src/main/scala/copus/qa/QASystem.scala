package copus.qa

import copus.corenlp._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.{SparkConf, SparkContext}

object QASystem {

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\winutils")
    val sparkConf = new SparkConf().setAppName("QASystemLab3").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)


    val hashingTF = new HashingTF()

    val questions = List (
      new Question("Why is it a good idea to specify object invariants in Python?", hashingTF),
      new Question("What makes an abstract data type abstract?", hashingTF),
      new Question("Are there any advantages to using postfix?", hashingTF),
      new Question("Does python provide a split method for tokenizing strings?", hashingTF),
      new Question("Can a module contain more than one function with the same name?", hashingTF),
      new Question("What is a string slice?", hashingTF),
      new Question("What is an accumulator?", hashingTF),
      new Question("What is a type conversion?", hashingTF)
    )


    //val documents = sc.objectFile[Document]("preparedCorpus\documents.obj")
    val sentences = sc.objectFile[Sentence]("preparedCorpus\\sentences.obj")

    questions.foreach(question => {
      val bestSentences = sentences.map(sentence => Tuple2(sentence, sentence.score(question)))
        .filter(tup => tup._2 > 0.0)
        .sortBy(tup => tup._2, ascending = false)
        .take(5)
      val candidateAnswers = bestSentences.map(sentence => sentence._1.extractAnswer(question))
      println("\n\nQuestion: " + question)
      candidateAnswers.foreach(answer => println("\nCandidate Answer: " + answer))

//      val bestDocs = documents.map(document => Tuple2(document, CoreNlpWrapper.scoreDocument(document, question)))
      //        .filter(tup => tup._2 > 0.0)
      //        .sortBy(tup => tup._2, ascending = false)
      //        .take(2)
      //
      //      val sentences = bestDocs.flatMap(tup => tup._1.getSentences)
      //      val bestSentences = sentences.map(sentence => Tuple2(sentence, CoreNlpWrapper.scoreSentence(sentence, question)))
      //        .filter(tup => tup._2 > 0.0)
      //        .sortBy(tup => tup._2).takeRight(10)
      //
      //      val candidateAnswers = bestSentences.map(sentence => sentence._1.extractAnswer(question))
      //      val bestAnswer = candidateAnswers.reduce(
      //        (a, b) => if (a.getAnswerQuality > b.getAnswerQuality) a else b
      //      )
//
//      println("\n\nQuestion: " + question.getText + "\nAnswer: " + bestAnswer)
    })
  }//main()

}//object QASystem
