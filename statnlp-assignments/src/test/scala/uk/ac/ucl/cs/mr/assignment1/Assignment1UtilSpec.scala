package uk.ac.ucl.cs.mr.assignment1

import org.scalatest.{Matchers, WordSpec}
import uk.ac.ucl.cs.mr.assignment1.Assignment1Util.{Counts, NGram, ngramsInSentence, getNGramCounts}
import ml.wolfe.nlp.{SentenceSplitter, TokenSplitter, Token, Sentence, Document}

/**
 * Created by williamferreira on 15/10/2014.
 **/
class Assignment1UtilSpec extends WordSpec with Matchers {
  "An addNgramCount" should {
    "add a new ngram count with default value for a new key" in {
      val counts: Assignment1Util.Counts = Map()
      val ngram: Assignment1Util.NGram = Seq[String]()
      Assignment1Util.addNgramCount(counts, ngram) shouldBe Map(ngram -> 1.0)
    }
  }

  "An addNgramCount" should {
    "add a new ngram count with given value for a new key" in {
      val counts: Assignment1Util.Counts = Map()
      val ngram: Assignment1Util.NGram = Seq[String]()
      Assignment1Util.addNgramCount(counts, ngram, 10.0) shouldBe Map(ngram -> 10.0)
    }
  }

  "An addNgramCount" should {
    "Update the count for an existing ngram" in {
      val ngram: Assignment1Util.NGram = Seq[String]()
      val counts: Assignment1Util.Counts = Map(ngram -> 1.0)

      Assignment1Util.addNgramCount(counts, ngram, 4.0) shouldBe Map(ngram -> 5.0)
    }
  }

  "An addNgramCounts" should {
    "Combine the counts in two ngram maps" in {
      val countsMany: Counts = Map(Seq("a", "b") -> 1.0, Seq("c", "d") -> 2.0, Seq("e") -> 3.0)
      val countsFew: Counts = Map(Seq("a", "b") -> 1.0, Seq("c", "d") -> 2.0, Seq("f") -> 10.0)

      (Assignment1Util.addNgramCounts(countsMany, countsFew) shouldBe
        Map(Seq("a", "b") -> 2.0, Seq("c", "d") -> 4.0, Seq("e") -> 3.0, Seq("f") -> 10.0))
    }
  }

  def tokenize(text: String): Sentence = (SentenceSplitter andThen TokenSplitter)(text).sentences(0)

  "An ngramsInSentence" should {
    "Find all the ngrams of a given size in a Sentence" in {
      val sentence = tokenize("abc def ghi abc def")

      val n = 2
      ngramsInSentence(sentence, n) shouldBe (
        Seq(Seq("abc", "def"), Seq("def", "ghi"), Seq("ghi", "abc"), Seq("abc", "def")))
    }
  }

  "An ngramsInSentence" should {
    "Return an empty sequence when the number of tokens is < n" in {
      val sentence = tokenize("abc def ghi abc def")

      val n = 6
      ngramsInSentence(sentence, n) shouldBe Seq[NGram]()
    }
  }

  "An ngramsInSentence" should {
    "Return an empty sequence when n=0" in {
      val sentence = tokenize("abc def ghi abc def")

      val n = 0
      ngramsInSentence(sentence, n) shouldBe Seq[NGram]()
    }
  }

  "A getNGramCounts" should {
    "Return a count of the ngrams in a Document" in {
      val sentence1 = tokenize("abc def ghi abc def")
      val sentence2 = tokenize("abc def ghi abc def")
      val sentence3 = tokenize("abc def ghi abc def")
      val document = Document("Test doc", Seq(sentence1, sentence2, sentence3))

      var n = 2
      getNGramCounts(document, n) shouldBe Map(
        Seq("abc", "def") -> 6.0, Seq("def", "ghi") -> 3.0, Seq("ghi", "abc") -> 3.0)

      n = 1
      getNGramCounts(document, n) shouldBe Map(
        Seq("abc") -> 6.0, Seq("def") -> 6.0, Seq("ghi") -> 3.0)
    }
  }
}