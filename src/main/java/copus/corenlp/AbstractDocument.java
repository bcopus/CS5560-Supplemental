package copus.corenlp;

import org.apache.spark.mllib.linalg.SparseVector;
import scala.Tuple2;

import java.io.Serializable;
import java.util.Arrays;

public abstract class AbstractDocument implements Serializable {
  public static final int NO_TOPIC = -1;
  
  protected Token[] tokens;
  protected Ngram2[] ngrams;
  protected SparseVector tfIdfVector;
  static protected double NGRAM_MATCH_MULTIPLIER = 1.3;
  static protected double LDA_TOPIC_MATCH_MULTIPLIER = 1.25;
  protected int mainTopic = NO_TOPIC;
  
  public Token[] getTokens() {
    return tokens;
  }
  public Ngram2[] getNgrams() { return ngrams; }
  
  public SparseVector getTfIdfVector() {
    return tfIdfVector;
  }
  
  public void setTfIdfVector(SparseVector tfIdfVector) {
    this.tfIdfVector = tfIdfVector;
  }
  
  public double score(Question question) {
    double score = 0.0;
    int[] indices = this.getTfIdfVector().indices();
    double[] values = this.getTfIdfVector().values();
    
    for(int hash : question.getTargetTermHashes()) {
      int index = Arrays.binarySearch(indices, hash);
      if(index >= 0)
        score += values[index];
    }
    
    for(int i = 0; i < question.getTargetNgrams().length; i++)
      for(int j = 0; j < ngrams.length; j++)
        if(ngrams[j].matchesNgram2(question.getTargetNgrams()[i]))
          score *= NGRAM_MATCH_MULTIPLIER;
    if(mainTopic != NO_TOPIC)
      if(mainTopic == question.mainTopic())
        score *= LDA_TOPIC_MATCH_MULTIPLIER;
    return score;
  }//score()
  
  public abstract void setMainTopicFromLDA(int i);
  
  public int mainTopic() { return mainTopic; }
  
  
}//class AbstractDocument
