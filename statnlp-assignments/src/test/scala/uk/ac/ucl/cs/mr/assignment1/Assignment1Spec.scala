package uk.ac.ucl.cs.mr.assignment1

import ml.wolfe.nlp.{Document, Sentence, TokenSplitter, SentenceSplitter}
import org.scalatest.{Matchers, WordSpec}
import uk.ac.ucl.cs.mr.assignment1.Assignment1._
import uk.ac.ucl.cs.mr.assignment1.Assignment1Util._

/**
 * Created by rockt on 08/10/2014.
 */
class Assignment1Spec extends WordSpec with Matchers {
  def tokenize(text: String): Seq[Sentence] = (SentenceSplitter andThen TokenSplitter)(text).sentences

  "A tokenizer" should {
    "should split a text into words" in {
      tokenize("This is a test.").flatMap(_.tokens).map(_.word) shouldBe List("This", "is", "a", "test", ".")
    }
  }

  // todo: additional unit tests go here

  "A ConstantLM" should {
    "define a LM in which the probability of a word is uniform over the vocabulary size" in {
      val clm = new ConstantLM(100)
      clm.prob("", Seq[String]()) shouldEqual 1.0/100
    }
  }

  val document = Document("Test doc", Seq.concat(
    tokenize("I am Sam"),
    tokenize("Sam I am"),
    tokenize("I do not eat green eggs and ham"))
  )

  val vocab = Map("I" -> 3, "am" -> 2, "Sam" -> 2, "do" -> 1,
                  "not" -> 1, "eat" -> 1, "green" -> 1, "eggs" -> 1,
                  "and" -> 1, "ham" -> 1)

  "An unigramLM" should {
    "Return the probabilities of unigrams" in {
      val order = 1
      val unigramLM = trainNgramLM(Seq(document), order)

      unigramLM.prob("<s>", Seq()) shouldEqual 3.0 / 20 +- 1e-2
      unigramLM.prob("I", Seq()) shouldEqual 3.0 / 20 +- 1e-2
      unigramLM.prob("am", Seq()) shouldEqual 2.0 / 20 +- 1e-2
    }
  }

  "An bigramLM" should {
    "Return the probabilities of bigrams" in {
      val order = 2
      val bigramLM = trainNgramLM(Seq(document), order)

      bigramLM.prob("I", Seq("<s>")) shouldEqual 0.67 +- 1e-2
      bigramLM.prob("Sam", Seq("<s>")) shouldEqual 0.33 +- 1e-2
      bigramLM.prob("am", Seq("I")) shouldEqual 0.67 +- 1e-2
      bigramLM.prob("</s>", Seq("Sam")) shouldEqual 0.5 +- 1e-2
      bigramLM.prob("Sam", Seq("am")) shouldEqual 0.5 +- 1e-2
      bigramLM.prob("do", Seq("I")) shouldEqual 0.33 +- 1e-2
    }
  }

  "Calling perplexity of on a constant LM should" should {
    "Return the size of the vocabulary" in {
      val vocabSize = 10
      val lm = new ConstantLM(vocabSize)
      val doc = Document("Test doc", tokenize("0 0 0 0 0 0 0 0 0 9"))

      perplexity(lm, Seq(doc).iterator) shouldEqual vocabSize.toDouble +- 1e-2
    }
  }

  class SkewedDigitLM() extends LanguageModel {
    val order = 1

    def prob(word: String, history: NGram) = {
      word match {
        case "0" => 10.0 / 19.0
        case _ => 1.0 / 19
      }
    }
  }

  "Calling perplexity of on a skewed digit LM should" should {
    "Return a prescribed value of approx 2.39" in {
      val lm = new SkewedDigitLM()
      val doc = Document("Test doc", tokenize("0 0 0 0 0 0 0 0 0 9"))

      perplexity(lm, Seq(doc).iterator) shouldEqual 2.39 +- 1e-2
    }
  }

  "A higher order NGram model" should {
    "have a perplexity less than or equal to a lower order NGram model when " +
      "using same document for training and test" in {
      val unigramLM = trainNgramLM(Seq(document), 1)
      val bigramLM = trainNgramLM(Seq(document), 2)

      perplexity(unigramLM, Seq(document).iterator) >= perplexity(bigramLM, Seq(document).iterator)
    }
  }

  "A higher order NGram model with add one smoothing" should {
    "have a perplexity less than or equal to a lower order NGram model with add one smoothing when " +
      "using same document for training and test" in {
      val vocabSize = 14
      val unigramLMAddOne = new AddkLM(trainNgramLM(Seq(document), 1), vocab)
      val bigramLMAddOne = new AddkLM(trainNgramLM(Seq(document), 2), vocab)

      perplexity(unigramLMAddOne, Seq(document).iterator) >= perplexity(bigramLMAddOne, Seq(document).iterator)
    }
  }
}
