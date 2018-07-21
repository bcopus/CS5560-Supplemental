package copus.corenlp;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Matrix;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Question implements Serializable {
  private String text;
  private Annotation annotation;
  private Token[] tokens;
  private Ngram2[] ngrams;
  
  private InterrogativeType interrogativeType;
  private Token[] targetTerms;
  private Ngram2[] targetNgrams;
  private int[] targetTermHashes;
  
  private int mainTopic = AbstractDocument.NO_TOPIC;
  
  public Question(String text, HashingTF hasher) {
    this.text = text;
    annotation = CoreNlpPipeline.getPipeline().process(text);
    tokens = extractTokens();
    ngrams = extractNgrams();
    targetNgrams = ngrams; //for future use
    interrogativeType = extractInterrogativeType();
    targetTerms = extractTargetTerms();
    expandTargetTerms();
    targetTermHashes = extractTargetTermHashes(hasher);
    System.out.print("Target terms: ");
    for(Token tok : targetTerms)
      System.out.print(tok.getText() + " ");
  }//Question()
  
  private Token[] extractTokens() {
    List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
    Token[] tokensArray = new Token[tokens.size()];
    int i = 0;
    for(CoreLabel tok : tokens)
      tokensArray[i++] = new Token(tok);
    return tokensArray;
  }//extractTokens()
  
  private Ngram2[] extractNgrams() {
    List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
    if(tokens.size() < 2)
      return null;
    Ngram2[] ngramsArray = new Ngram2[tokens.size() - 1];
    for(int i = 0; i < ngramsArray.length; i++)
      ngramsArray[i] = new Ngram2(tokens.get(i), tokens.get(i + 1));
    return ngramsArray;
  }//extractTokens()
  
  private InterrogativeType extractInterrogativeType() {
    //  given a token with pos == "WP" or pos == "WRB", extract its lemma, and
    //  use it to determine the question type.
    for(Token tok : tokens) {
      if (tok.getPos() == PartOfSpeech.WP || tok.getPos() == PartOfSpeech.WRB) {
        switch (tok.getLemma().toLowerCase()) {
          case "who"   : return InterrogativeType.WHO;
          case "what"  : return InterrogativeType.WHAT;
          case "when"  : return InterrogativeType.WHEN;
          case "where" : return InterrogativeType.WHERE;
          case "why"   : return InterrogativeType.WHY;
        }//switch
      }
    }
    return InterrogativeType.UNKNOWN;
  }//extractInterrogativeType()
  
  private Token[] extractTargetTerms() {
    //collect the nouns and verbs from the question
    ArrayList<Token> targets = new ArrayList<>();
    for(Token tok : tokens)
      if(tok.isNoun() || tok.isVerb())
        targets.add(tok);
    return targets.toArray(new Token[targets.size()]);
  }//extractTargetTerms()
  
  private int[] extractTargetTermHashes(HashingTF hasher) {
    int[] hashes = new int[targetTerms.length];
    for(int i = 0; i < hashes.length; i++)
      hashes[i] = hasher.indexOf(targetTerms[i].getLemma());
    return hashes;
  }//extractTargetTermHashes()
  
  public String getText() {
    return text;
  }
  
  public Token[] getTokens() {
    return tokens;
  }
  
  public InterrogativeType getInterrogativeType() {
    return interrogativeType;
  }
  
  public Token[] getTargetTerms() {
    return targetTerms;
  }
  
  public int[] getTargetTermHashes() {
    return targetTermHashes;
  }
  
  public Ngram2[] getNgrams() {
    return ngrams;
  }
  
  public Ngram2[] getTargetNgrams() {
    return targetNgrams;
  }

  private void expandTargetTerms() {
    ArrayList<Token> targets = new ArrayList<>();
    targets.addAll(Arrays.asList(targetTerms));
    for(Token tt : targetTerms) {
      String[] exp = Models.expand(tt.getLemma());
      if(exp != null) {
        for(String s : exp) {
          Token t = new Token();
          t.setLemma(s);
          t.setText(s);
          t.setNec(tt.getNec());
          t.setPos(tt.getPos());
          targets.add(t);
        }
      }
    }
    targetTerms = new Token[targets.size()];
    targets.toArray(targetTerms);
  }//expandTargetTerms()
  
 
  public int mainTopic() { return mainTopic; }
  public void setMainTopic(int topic) { mainTopic = topic; }
  
  @Override
  public String toString() {
    return "Question{" +
            "text='" + text + '\'' +
            "main topic=" + mainTopic +
            '}';
  }
}//class Question
