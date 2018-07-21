package copus.corenlp;

public class Answer {
  private String answerText;
  private double answerQuality;
  
  static public double QUALITY_VERY_HIGH = 1.0;
  static public double QUALITY_HIGH = 0.75;
  static public double QUALITY_MEDIUM = 0.5;
  static public double QUALITY_LOW = 0.25;
  static public double QUALITY_VERY_LOW = 0.0;
  
  public Answer(String answerText, double answerQuality) {
    this.answerText = answerText;
    this.answerQuality = answerQuality;
  }
  
  public String getAnswerText() {
    return answerText;
  }

    public double getAnswerQuality() {
    return answerQuality;
  }
  
  public Answer setAnswerText(String answerText) {
    this.answerText = answerText;
    return this;
  }
  
  public Answer setAnswerQuality(double answerQuality) {
    this.answerQuality = answerQuality;
    return this;
  }
  
  @Override
  public String toString() {
    return "(" + answerQuality + ") " + answerText;
  }

}//class Answer
