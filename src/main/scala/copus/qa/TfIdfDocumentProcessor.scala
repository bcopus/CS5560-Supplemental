package copus.qa

import java.io.PrintStream
import java.io._

import copus.corenlp._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.mllib.clustering.{EMLDAOptimizer, LDA, LDAModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.commons.io.FileUtils
import scala.collection.mutable.ListBuffer

object TfIdfDocumentProcessor {

  def processElements(rDD: RDD[AbstractDocument]): Seq[AbstractDocument] = {
    //process each document into a sequence of lemmatized tokens
    val elementSequences =
      rDD.map(doc => {
        doc.getTokens.map(tok => {
          tok.getLemma
        }).toSeq
      })
    elementSequences.foreach(doc => println(doc))

    //calculate TF-IDF data
    val hashingTF = new HashingTF()
    val termFrequencies = hashingTF.transform(elementSequences)
    termFrequencies.cache()

    val idf = new IDF().fit(termFrequencies) //returns IDFModel
    val tfIdf = idf.transform(termFrequencies) //returns TF-IDF SparseVectors

    //collect elements and TF-IDF data and save the vectors in each element
    val dd = rDD.collect()
    val tt = tfIdf.collect()

    for (i : Int <- 0 until rDD.count().toInt) {
      dd.apply(i).setTfIdfVector(tt.apply(i).toSparse)
    }
    return dd
  }//processElements()

  private case class Params(
                             input: Seq[AbstractDocument] = Seq.empty,
                             k: Int = 20,
                             algorithm: String = "em")

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\winutils")
    val sparkConf = new SparkConf().setAppName("QASystem").setMaster("local[*]")
    val sc: SparkContext = new SparkContext(sparkConf)

    FileUtils.deleteDirectory(new File("preparedCorpus"))
    new File("preparedCorpus").mkdir()

    val files = sc.wholeTextFiles("corpus")

    //val stops : Broadcast[Array[String]] =sc.broadcast(sc.textFile("data/stopwords.txt").collect())

    //
    //Process the corpus documents with full annotation;
    //  Save the TF-IDF data per-document and per-sentence;
    //  Save documents and sentences in object files for subsequent use.
    //    First: The documents
    //
    val documents: Array[Document] = files.map(file => CoreNlpWrapper.prepareText(file._2)).toArray()
    val docsRDD : RDD[AbstractDocument] = sc.parallelize(documents)

    //
    //      Calculate TF-IDF for documents and save the vector in each document
    //
    val dd: Seq[AbstractDocument] = processElements(docsRDD)

    //
    //    Second: the sentences
    //
    val sentences = documents.flatMap(doc => doc.getSentences)
    val sentencesRDD : RDD[AbstractDocument] = sc.parallelize(sentences)

    val ss: Seq[AbstractDocument] = processElements(sentencesRDD)
    //defer saving sentenced object file until LDA topic extraction has been performed

    //
    //...Done with TF/IDF processing
    //

    //
    //Use Word2Vec to create a vector space from the corpus for subsequent use
    //
    val w2vecInput : RDD[Seq[String]] = sc.parallelize(sentences.map(s => s.getTokens.map(t => t.getLemma).toSeq).toSeq)
    val word2vec : Word2Vec = new Word2Vec().setVectorSize(300)
    val model: Word2VecModel = word2vec.fit(w2vecInput)
    model.save(sc, "preparedCorpus/w2vec")

    //
    //Prepare for LDA topic extraction
    //

    val params: Params = Params(ss, 20, "em")

    val topic_output: PrintStream = new PrintStream("preparedCorpus\\LDAResults.txt")
    val (corpus, vocabArray, actualNumTokens) = preprocess(sc, params.input)
    corpus.cache()
    val actualVocabSize = vocabArray.length

    //
    // Run LDA topic extraction
    //
    val lda = new LDA()

    lda.setOptimizer(new EMLDAOptimizer)
      .setK(params.k)
      .setMaxIterations(20)

    //val cvm : CountVectorizerModel = new CountVectorizerModel(corpus)
    val ldaModel: LDAModel = lda.run(corpus)

    //
    // Save the LDA model for later use by the QA system
    //
    ldaModel.save(sc, "preparedCorpus\\LDA")
    sc.parallelize(vocabArray).saveAsObjectFile("preparedCorpus\\vocab.obj")

    topic_output.println(s"Finished training LDA model.  Summary:")

    //
    // Print the LDA topics, showing the top-weighted terms for each topic.
    //
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = actualVocabSize)
    val topics = topicIndices.map { case (terms, termWeights) =>
      terms.zip(termWeights).map { case (term, weight) => (vocabArray(term.toInt), weight) }
    }
    topic_output.println(s"${params.k} topics:")
    topics.zipWithIndex.foreach { case (topic, i) =>
      topic.foreach { case (term, weight) =>
        println(s"$term\t$weight")
        topic_output.println(s"TOPIC_$i;$term;$weight")
      }
      topic_output.println()
    }
    topic_output.close()

    //
    // Apply LDA to extract best topics for each sentence, and save the sentences and document
    // to object files
    //
    Models.setLDAModel(ldaModel)
    //Models.setVocabulary(vocabArray)
    val ti = ldaModel.describeTopics(maxTermsPerTopic = math.min(actualVocabSize, 50))
    val topWords: Array[Array[String]] = ti.map {
      case (terms, termWeights) => terms.zip(termWeights).map {
        case (term, weight) => vocabArray(term.toInt)
      }
    }

    extractBestTopics(sc, ss, topWords)
    val tw: Seq[Seq[String]] = topWords.map(topic => topic.toSeq).toSeq
    sc.parallelize(tw).saveAsObjectFile("preparedCorpus\\ldaTopicsTopWords.obj")
    sc.parallelize(ss).saveAsObjectFile("preparedCorpus\\sentences.obj")
    sc.parallelize(dd).saveAsObjectFile("preparedCorpus\\documents.obj")
    //
    // End of LDA processing
    //

  }//main()

  private def preprocess(sc: SparkContext, docs: Seq[AbstractDocument]): (RDD[(Long, Vector)], Array[String], Long) = {
    val stopWords: Array[String] =sc.textFile("data/stopwords.txt").collect()
    val stopWordsBroadCast: Broadcast[Array[String]] =sc.broadcast(stopWords)

    val filteredDocs: RDD[Array[String]] = sc.parallelize(
      docs.map(
      (doc: AbstractDocument) => doc.getTokens.map((tok: Token) => tok.getLemma)
        .filterNot((lemma: String) => stopWordsBroadCast.value.contains(lemma.toLowerCase))
        .filterNot((lemma: String) => lemma.contains("[^a-zA-Z]".r))
        .filterNot((lemma : String) => lemma.length <= 3)
      )
    )
    filteredDocs.cache()

    val dfseq: RDD[Seq[String]] = filteredDocs.map(_.toSeq)
    dfseq.cache()

    val hashingTF = new HashingTF(filteredDocs.count().toInt)
    val tf = hashingTF.transform(dfseq)
    tf.cache()

    val idf = new IDF().fit(tf)
    val tfidf = idf.transform(tf).zipWithIndex().map(_.swap)

    val dff = dfseq.flatMap(f => f)
    val vocab: Array[String] = dff.distinct().collect()

    (tfidf, vocab, dff.count()) // Vector, Vocab, total token count

  }//preprocess

  private def extractBestTopics(sc: SparkContext, ss: Seq[AbstractDocument], topWords: Array[Array[String]]): Unit = {
    ss.foreach((sentence: AbstractDocument) => {
      val topicAffinities = ListBuffer[Int]()
      val lemmas = sentence.getTokens.map((tok: Token) => tok.getLemma)
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
      sentence.setMainTopicFromLDA(bestTopic)
    })
  }

}//object TfIdfDocumentProcessor
