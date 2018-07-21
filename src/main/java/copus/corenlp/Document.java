package copus.corenlp;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;
import org.apache.spark.mllib.linalg.SparseVector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

public class Document extends AbstractDocument implements Serializable {
  private Sentence[] sentences;

  public Document(Annotation doc) {
    //Produce sentences
    ArrayList<Sentence> ss = new ArrayList<>();
    for (CoreMap sentence : doc.get(CoreAnnotations.SentencesAnnotation.class)) {
     ss.add(new Sentence(sentence));
    }
    sentences = new Sentence[ss.size()];
    ss.toArray(sentences);
    
    //Produce Tokens and N-grams
    ArrayList<Token> tokens = new ArrayList<>();
    for(Sentence s : sentences) {
      tokens.addAll(Arrays.asList(s.getTokens()));
    }
    this.tokens = new Token[tokens.size()];
    tokens.toArray(this.tokens);
    
    tfIdfVector = null;
  }//Document()
  
  public Sentence[] getSentences() {
    return sentences;
  }
  
 
  @Override
  public void setMainTopicFromLDA(int topicIndex) {
    mainTopic = topicIndex;
  }
  
  @Override
  public String toString() {
    StringBuffer sb = new StringBuffer();
    sb.append(tfIdfVector == null ? "{no TF-IDF} " : "{TF-IDF set} ");
    for(Sentence sentence : sentences)
      sb.append(sentence + "\n");
    return sb.toString();
  }//toString();
}//class Document
