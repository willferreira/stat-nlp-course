package uk.ac.ucl.cs.mr.assignment1

import ml.wolfe.nlp.{Token, TokenSplitter, SentenceSplitter}
import org.scalatest.{Matchers, WordSpec}
import uk.ac.ucl.cs.mr.assignment1.{Assignment1}
import uk.ac.ucl.cs.mr.assignment1.{Assignment1Util}

/**
 * Created by rockt on 08/10/2014.
 */
class Assignment1Spec extends WordSpec with Matchers {
  def tokenize(text: String): Seq[Token] = (SentenceSplitter andThen TokenSplitter)(text).sentences.flatMap(_.tokens)

  "A tokenizer" should {
    "should split a text into words" in {
      tokenize("This is a test.").map(_.word) shouldBe List("This", "is", "a", "test", ".")
    }
  }

  //todo: additional unit tests go here

  "A ConstantLM" should {
    "define a LM in which the probability of a word is uniform over the vocabulary size" in {
      val clm = new Assignment1.ConstantLM(100)
      clm.prob("", Seq[String]()) shouldEqual 1.0/100
    }
  }
}
