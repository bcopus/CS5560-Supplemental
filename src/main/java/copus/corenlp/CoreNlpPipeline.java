package copus.corenlp;

import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import java.util.Properties;

class CoreNlpPipeline {
  
  private static StanfordCoreNLP pipeline = null;
  
  private  CoreNlpPipeline() {}
  
  static StanfordCoreNLP getPipeline() {
    if(pipeline == null) {
      Properties props = new Properties();
      props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner");
      pipeline = new StanfordCoreNLP(props);
    }
    return pipeline;
  }//getPipeline()
}//class CoreNlpPipeline
