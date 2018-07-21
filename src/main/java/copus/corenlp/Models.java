package copus.corenlp;

import org.apache.spark.mllib.clustering.LDAModel;
import org.apache.spark.mllib.feature.Word2VecModel;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.HashMap;

public class Models {
  private static Word2VecModel w2vModel;
  private static LDAModel ldaModel;
  private static String[][] ldaTopWords;
  //private static String[] vocabulary;
  

  public static void setW2VModel(Word2VecModel model) { w2vModel = model; }
  public static void setLDAModel(LDAModel model) { ldaModel = model; }
  public static void setLDATopWords(String[][] words) { ldaTopWords = words; }
  //public static void setVocabulary(String[] vocab) { vocabulary = vocab; }

  public static String[] expand(String word) {
    Tuple2<String, Object>[] synonyms;
    try {
      synonyms = w2vModel.findSynonyms(word, 3);
    } catch (java.lang.IllegalStateException ex) {
      return null;
    }
    String[] rval = new String[synonyms.length];
    for(int i = 0; i < rval.length; i++) {
      rval[i] = synonyms[i]._1;
      //System.out.println("Syn: " + word + ", " + rval[i]);
    }
    return rval;
  }//expand()
  
  public static LDAModel ldaModel() { return ldaModel; }
  public static String[][] ldaTopWords() { return ldaTopWords; }
//  public static String[] vocabulary() { return vocabulary; }

}//class WordExpansion
