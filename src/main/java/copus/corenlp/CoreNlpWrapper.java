package copus.corenlp;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class CoreNlpWrapper {
  
  static Random random = new Random();
  
  private static StanfordCoreNLP getPipeline() {
    return CoreNlpPipeline.getPipeline();
  }
  
  public static Document prepareText(String text) {
    return new Document(getPipeline().process(text));
  }//prepareText()
  
  public static void initializePipeline() {
    getPipeline();
  }
    
}//class CoreNlpWrapper
