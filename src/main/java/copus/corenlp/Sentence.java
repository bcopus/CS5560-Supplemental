package copus.corenlp;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.CoreMap;
import org.apache.spark.mllib.linalg.SparseVector;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

public class Sentence extends AbstractDocument implements Serializable, AnswerProvider {
  private String originalText;
  
  public Sentence(CoreMap sentence) {
    originalText = sentence.get(CoreAnnotations.TextAnnotation.class);
    
    ArrayList<Token> ts = new ArrayList<>();
    ArrayList<Ngram2> ng = new ArrayList<>();
    
    CoreLabel lastTok = null;
    for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
      ts.add(new Token(token));
      if(lastTok != null)
        ng.add(new Ngram2(lastTok, token));
      lastTok = token;
    }
    
    tokens = new Token[ts.size()];
    ts.toArray(tokens);
  
    ngrams = new Ngram2[ng.size()];
    ng.toArray(ngrams);
    
  }//Sentence
  
  @Override
  public void setMainTopicFromLDA(int topicIndex) {
    mainTopic = topicIndex;
    System.out.println("TOPIC SELECTED: " + mainTopic);
  }//setMainTopicFromLDA
  
  public String getOriginalText() {
    return originalText;
  }
  
  public Answer extractAnswer(Question question) {
    //construct the default answer to return if we cannot improve upon it
    Answer answer = new Answer(this.getOriginalText(), Answer.QUALITY_VERY_LOW);
  
    //For "why" questions, look for a "because" clause...
    if(question.getInterrogativeType() == InterrogativeType.WHY) {
      Token[] tokens = this.getTokens();
      int becauseIndex = -1;
      for (int i = 0; i < tokens.length; i++) {
        if(tokens[i].getPos()==PartOfSpeech.IN && tokens[i].getLemma().toLowerCase().equals("because"))
          becauseIndex = i;
      }
      if(becauseIndex >= 0) {
        StringBuffer sb = new StringBuffer();
        for (int i = becauseIndex; i < tokens.length; i++) {
          sb.append(tokens[i].getText() + " ");
        }
        answer.setAnswerText(sb.toString()).setAnswerQuality(Answer.QUALITY_VERY_HIGH);
      }
      return answer;
    }
    //For other question types, look for an appropriate named entity...
    ArrayList<NamedEntityClass> objectives = new ArrayList<>();
    switch (question.getInterrogativeType()) {
      case WHO   :
      case WHAT  :
        objectives.add(NamedEntityClass.PERSON);
        objectives.add(NamedEntityClass.ORGANIZATION);
        break;
      case WHEN  :
        objectives.add(NamedEntityClass.DATE);
        objectives.add(NamedEntityClass.TIME);
        break;
      case WHERE :
        objectives.add(NamedEntityClass.LOCATION);
        break;
      default: return answer; //no idea what to extract, use default answer
    }
  
    //we have at least one target class to extract
    NamedEntityClass[] targets = new NamedEntityClass[objectives.size()];
    objectives.toArray(targets);
    Token[] tokens = this.getTokens();
    int start = -1, end = -1;
    for (int i = 0; i < tokens.length; i++) {
      for (int j = 0; j < targets.length; j++) {
        if(tokens[i].getNec() == targets[j]) {
          if (start < 0) {
            start = i;
          }
          end = i;
        }
      }//inner for
    }//outer for
  
    //if no matching named entities found, and it is a WHAT question, seek a definition
    if(start == -1) {
      for(int i = 0; i < tokens.length; i++) {
        if(question.getInterrogativeType()==InterrogativeType.WHAT
                && (tokens[i].getLemma().equals(":") || tokens[i].getLemma().equals("be")))
          start = i < tokens.length - 1 ? i + 1 : -1;
      }
      if(start > -1)
        end = tokens.length - 1;
    }

    //if nothing found, return the default answer
    if(start == -1)
      return answer;
  
    //otherwise, extract the named entity tokens and indicate a high quality answer
    StringBuffer sb = new StringBuffer();
    for(int i = start; i < end; i++)
      sb.append(tokens[i].getText() + " ");
    return answer.setAnswerText(sb.toString()).setAnswerQuality(Answer.QUALITY_VERY_HIGH);
  }//extractAnswer()
  
  @Override
  public String toString() {
    StringBuffer sb = new StringBuffer();
    sb.append("<" + originalText + "> ");
    for(Token token : tokens)
      sb.append(token + " ");
    return sb.toString();
  }//toString();
}//Sentence
