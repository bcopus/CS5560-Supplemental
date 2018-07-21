package copus.corenlp;

public enum NamedEntityClass {
  PERSON, LOCATION, ORGANIZATION, MISC,   //named classes
  MONEY, NUMBER, ORDINAL, PERCENT,        //numeric classes
  DATE, TIME, DURATION, SET,              //temporal classes
  NONE                                    //none of the above
}
