package copus.qa

import java.io.File
import javafx.application.Application
import javafx.event.{ActionEvent, EventHandler}
import javafx.geometry.Pos
import javafx.scene.Scene
import javafx.scene.control.{Button, Label, TextArea, TextField}
import javafx.scene.layout.GridPane
import javafx.scene.text.Font
import javafx.stage.Stage

import copus.corenlp._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.clustering.DistributedLDAModel
import scala.collection.mutable.ListBuffer

object fxQA {
  def main(args: Array[String]) {
    Application.launch(classOf[fxQA], args: _*)
  }
}//object fxQA

class fxQA extends Application {
  var question : String = ""

  override def start(primaryStage: Stage) {
    //
    //Initialize Spark and the Stanford CoreNLP Pipeline
    //
    System.setProperty("hadoop.home.dir", "C:\\winutils")
    val sparkConf = new SparkConf().setAppName("QASystemLab3").setMaster("local[*]")
    val sc = new SparkContext(sparkConf)
    CoreNlpWrapper.initializePipeline()

    //
    //Load the corpus
    //
    val hashingTF = new HashingTF()
    val sentences: RDD[Sentence] = sc.objectFile[Sentence]("preparedCorpus\\sentences.obj")
    val documents: RDD[Sentence] = sc.objectFile[Sentence]("preparedCorpus\\documents.obj")

    //
    //Load the Word2Vec Model
    //
    val w2vFolderPath : String = "preparedCorpus\\w2vec"
    val w2vModelFolder = new File(w2vFolderPath)
    if (!w2vModelFolder.exists())
      throw new RuntimeException("No Word2Vec model exists at path=" + w2vFolderPath)
    val w2VModel = Word2VecModel.load(sc, w2vFolderPath)
    Models.setW2VModel(w2VModel)

    //
    //Load the LDA Model
    //
    val ldaFolderPath : String = "preparedCorpus\\LDA"
    val ldaModelFolder = new File(ldaFolderPath)
    if(!ldaModelFolder.exists())
      throw new RuntimeException("No LDA model exists at path=" + ldaFolderPath)
    val ldaModel: DistributedLDAModel = DistributedLDAModel.load(sc, ldaFolderPath)
    Models.setLDAModel(ldaModel)

    val topWords: Array[Array[String]] = sc.objectFile[Seq[String]]("preparedCorpus\\ldaTopicsTopWords.obj")
        .collect().map(topic => topic.toArray)

    Models.setLDATopWords(topWords)

    //
    //Initialize GUI
    //
    initializeGui(sc, primaryStage, hashingTF, sentences)

  }//start

  def answerQuestion(sc : SparkContext, qtext : String,
                     hashingTF: HashingTF, sentences: RDD[Sentence]
                    ) : String = {
    val question = new Question(qtext, hashingTF) //expands question using WordToVec model
    annotateMainTopic(sc, question, Models.ldaTopWords())
    val bestSentences: Array[(Sentence, Double)] = sentences.map(sentence => Tuple2(sentence, sentence.score(question)))
      .filter(tup => tup._2 > 0.0)
      .sortBy(tup => tup._2, ascending = false)
      .take(5)
    val answers : Seq[Answer] = bestSentences.map(sentence => sentence._1.extractAnswer(question))
    answers.map(ans => ans.getAnswerText).reduce((a, b) => a + "\n---------------------------------\n" + b)
  }//answerQuestion

  def annotateMainTopic(sc : SparkContext, q : Question, topWords : Array[Array[String]]): Unit = {
    val topicAffinities = ListBuffer[Int]()
    val lemmas = q.getTokens.map((tok: Token) => tok.getLemma)
    topWords.foreach(topic => {
      val accum = sc.accumulator(0)
      lemmas.foreach(lemma => {
        topic.foreach( keyWord =>
          if(keyWord == lemma)
            accum += 1
        )
      })
      topicAffinities += accum.value
    })
    val bestTopic = {
      if (topicAffinities.forall(score => score == 0))
        AbstractDocument.NO_TOPIC
      else
        topicAffinities.zipWithIndex.maxBy(_._1)._2
    }
    q.setMainTopic(bestTopic)
  }//annotateMainTopic

  def initializeGui(sc : SparkContext, primaryStage: Stage,
                    hashingTF: HashingTF, sentences: RDD[Sentence]
                   ): Unit = {
    primaryStage.setTitle("Copus QA")

    val pane: GridPane = new GridPane
    val btOK: Button = new Button ("OK")
    val tfQuestion: TextField = new TextField()
    val tfAnswer: TextArea = new TextArea
    val qlabel: Label = new Label("Question")
    val alabel: Label = new Label("Possible Answers")
    val font: Font = Font.font("Verdana", 20.0)

    pane.setAlignment(Pos.CENTER)
    pane.add(btOK, 0, 6)
    pane.add(qlabel, 0, 0)
    pane.add(alabel, 0, 4)
    pane.add (tfQuestion, 0, 1)
    pane.add (tfAnswer, 0, 5)

    btOK.setOnAction(new EventHandler[ActionEvent] {
      override def handle(e: ActionEvent) {
        question = tfQuestion.getText ()
        //call the QA system here.
        tfAnswer.setText (answerQuestion(sc, question, hashingTF, sentences))
      }
    })

    primaryStage.setScene(new Scene(pane, 800, 400))
    primaryStage.show()
  }

}//class fxQA
