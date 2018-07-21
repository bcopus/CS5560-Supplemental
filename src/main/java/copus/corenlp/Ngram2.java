package copus.corenlp;

import edu.stanford.nlp.ling.CoreLabel;

import java.io.Serializable;

public class Ngram2 implements Serializable {
    private String ngram;

    public Ngram2(CoreLabel token1, CoreLabel token2) {
        ngram = token1 + " " + token2;
    }

    public boolean matchesNgram2(Ngram2 ngram2) {
        return this.ngram.equals(ngram2.ngram);
    }
}
