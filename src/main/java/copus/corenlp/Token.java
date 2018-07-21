package copus.corenlp;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;

import java.io.Serializable;

public class Token implements Serializable {
  private String text;
  private String lemma;
  private PartOfSpeech pos;
  private NamedEntityClass nec; //named entity class

  public Token() {  }
  
  public Token(CoreLabel tok) {
    text = tok.get(CoreAnnotations.TextAnnotation.class);
    lemma = tok.get(CoreAnnotations.LemmaAnnotation.class);
    pos = extractPos(tok.get(CoreAnnotations.PartOfSpeechAnnotation.class));
    nec = extractNec(tok.get(CoreAnnotations.NamedEntityTagAnnotation.class));
  }//Token()
  
  private PartOfSpeech extractPos(String pos) {
    //Penn Treebank POS tags: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    
    if(pos == null) return PartOfSpeech.UNLISTED;
    switch (pos) {
      case "CC"  : return PartOfSpeech.CC;
      case "CD"  : return PartOfSpeech.CD;
      case "DT"  : return PartOfSpeech.DT;
      case "EX"  : return PartOfSpeech.EX;
      case "FW"  : return PartOfSpeech.FW;
      case "IN"  : return PartOfSpeech.IN;
      case "JJ"  : return PartOfSpeech.JJ;
      case "JJR" : return PartOfSpeech.JJR;
      case "JJS" : return PartOfSpeech.JJS;
      case "LS"  : return PartOfSpeech.LS;
      case "MD"  : return PartOfSpeech.MD;
      case "NN"  : return PartOfSpeech.NN;
      case "NNS" : return PartOfSpeech.NNS;
      case "NNP" : return PartOfSpeech.NNP;
      case "NNPS": return PartOfSpeech.NNPS;
      case "PDT" : return PartOfSpeech.PDT;
      case "POS" : return PartOfSpeech.POS;
      case "PRP" : return PartOfSpeech.PRP;
      case "PRP$": return PartOfSpeech.PRP$;
      case "RB"  : return PartOfSpeech.RB;
      case "RBR" : return PartOfSpeech.RBR;
      case "RBS" : return PartOfSpeech.RBS;
      case "RP"  : return PartOfSpeech.RP;
      case "SYM" : return PartOfSpeech.SYM;
      case "TO"  : return PartOfSpeech.TO;
      case "UH"  : return PartOfSpeech.UH;
      case "VB"  : return PartOfSpeech.VB;
      case "VBD" : return PartOfSpeech.VBD;
      case "VBG" : return PartOfSpeech.VBG;
      case "VBN" : return PartOfSpeech.VBN;
      case "VBP" : return PartOfSpeech.VBP;
      case "VBZ" : return PartOfSpeech.VBZ;
      case "WDT" : return PartOfSpeech.WDT;
      case "WP"  : return PartOfSpeech.WP;
      case "WP$" : return PartOfSpeech.WP$;
      case "WRB" : return PartOfSpeech.WRB;
      default    :  return PartOfSpeech.UNLISTED;
    }//switch
  }//extractPos()
  
  private NamedEntityClass extractNec(String ner) {
    if(ner == null) return NamedEntityClass.NONE;
    switch (ner) {
      //named entity classes
      case "PERSON"       : return NamedEntityClass.PERSON;
      case "LOCATION"     : return NamedEntityClass.LOCATION;
      case "ORGANIZATION" : return NamedEntityClass.ORGANIZATION;
      case "MISC"         : return NamedEntityClass.MISC;  //misc. named class
      
      //numeric classes
      case "MONEY"        : return NamedEntityClass.MONEY;
      case "NUMBER"       : return NamedEntityClass.NUMBER;
      case "ORDINAL"      : return NamedEntityClass.ORDINAL;
      case "PERCENT"      : return NamedEntityClass.PERCENT;
      
      //temporal classes
      case "DATE"         : return NamedEntityClass.DATE;
      case "TIME"         : return NamedEntityClass.TIME;
      case "DURATION"     : return NamedEntityClass.DURATION;
      case "SET"          : return NamedEntityClass.SET;
      
      //no recognized entity class
      default             : return NamedEntityClass.NONE;
    }//switch
  }//extractNec()
  
  public boolean isNoun() {
    switch (pos) {
      case CD   :
      case NN   :
      case NNS  :
      case NNP  :
      case NNPS : return true;
      default   : return false;
    }//switch
  }//isNoun()
  
  public boolean isVerb() {
    switch (pos) {
      case VB  :
      case VBD :
      case VBG :
      case VBN :
      case VBP :
      case VBZ : return true;
      default  : return false;
    }//switch
  }//isNoun()
  
  public String getText() {
    return text;
  }
  
  public String getLemma() {
    return lemma;
  }

  public boolean lemmaMatches(Token tok) { return this.lemma.equals(tok.lemma); }
  
  public PartOfSpeech getPos() {
    return pos;
  }
  
  public NamedEntityClass getNec() {
    return nec;
  }

  public void setText(String text) {
    this.text = text;
  }

  public void setLemma(String lemma) {
    this.lemma = lemma;
  }

  public void setPos(PartOfSpeech pos) {
    this.pos = pos;
  }

  public void setNec(NamedEntityClass nec) {
    this.nec = nec;
  }

  @Override
  public String toString() {
    return "(" + text + ", " + lemma + ", " + pos + ", " + nec + ")";
  }//toString();
}//class Token
