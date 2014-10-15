package uk.ac.ucl.cs.mr.assignment1

import org.scalatest.{Matchers, WordSpec}
import uk.ac.ucl.cs.mr.assignment1.Assignment1Util.Counts
import uk.ac.ucl.cs.mr.assignment1.{Assignment1Util}

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

  //todo: additional unit tests go here
}