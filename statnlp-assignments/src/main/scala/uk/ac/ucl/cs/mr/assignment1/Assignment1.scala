package uk.ac.ucl.cs.mr.assignment1

import java.io.{File}
import scala.language.postfixOps
import scala.collection.immutable.Range
import scala.collection.mutable.Set

import ml.wolfe.nlp.Document

/**
 * @author Sebastian Riedel
 */
object Assignment1 {

  import uk.ac.ucl.cs.mr.assignment1.Assignment1Util._

  trait LanguageModel {
    def order: Int

    def prob(word: String, history: NGram): Double
  }

  /**
   * Calculate the perplexity of a language model, given a test set of documents.
   * @param lm the language mode
   * @param documents an iterator over a collection of Documents
   * @return a Double representing the perplexity measure
   */
  def perplexity(lm: LanguageModel, documents: Iterator[Document]): Double = {
    var s = 0.0
    var N = 0.0

    for (document <- documents) {
      for (sentence <- document.sentences) {
        val words = Seq.concat(
          Seq.fill(lm.order - 1) {
            "<s>"
          },
          sentence.tokens map (_.word),
          Seq.fill(lm.order - 1) {
            "</s>"
          }
        )

        // todo check sentence length in case it is too short for the language model

        // Jurafsky mentions not adding the number of end-of-sentence
        // markers to the word count N in the perplexity measure,
        // so remove number here
        N += words.length - lm.order + 1
        for (i <- Range(lm.order - 1, words.length)) {
          // compute the probability in logs and then convert
          // back at the end to avoid arithmetic unpleasantness
          val prob = lm.prob(words(i), words.slice(i - lm.order + 1, i))
          s += Math.log(prob)
        }
      }
    }
    Math.exp(-s / N)
  }

  class ConstantLM(vocabSize: Int) extends LanguageModel {
    val order = 1

    def prob(word: String, history: NGram) = 1.0 / vocabSize
  }

  class NgramLM(val countsN: Counts,
                val countsNMinus1: Counts,
                val order: Int) extends LanguageModel {
    def prob(word: String, history: NGram) = {
      (countsN getOrElse(history :+ word, 0.0)) / (countsNMinus1 getOrElse(history, 0.0))
    }
  }

  class AddOneLM(ngramLM: NgramLM, vocabSize: Int, eps: Double = 1.0) extends LanguageModel {
    def order = ngramLM.order

    def prob(word: String, history: NGram) = {
      (1 + (ngramLM.countsN getOrElse(history :+ word, 0.0))) / (vocabSize + (ngramLM.countsNMinus1 getOrElse(history, 0.0)))
    }
  }

  class GoodTuringLM(ngramLM: NgramLM) extends LanguageModel {
    def order = ngramLM.order

    def prob(word: String, history: NGram) = ???
  }

  def trainNgramLM(train:Seq[Document], order:Int): NgramLM = {
    val countsN = (train map (d => getNGramCounts(d, order))).foldLeft(Map[NGram, Double]())(addNgramCounts)
    new NgramLM(countsN, getNMinus1Counts(countsN), order)
  }

  /**
   * Generate the relevant folder names for the P series of the ACL Corpus, given a start and end year
   * @param start the start year
   * @param end the end year
   * @return a Collection of folder names
   */
  def getFolderNames(start: Int, end: Int) = {
    Range(start, end) map {i: Int => ("P" + (i.toString().takeRight(2)))}
  }

  // the location of the ACL Corpus data on my local disk
  val fileRoot = "/Users/williamferreira/Documents/UCL/CSML/Statistical NLP/Assignments/ACL/P/"

  /**
   * Load a corpus of data from the ACL Corpus, given the start and end year.
   * @param start the start year
   * @param end the end year
   * @return a tuple (Seq[String], Seq(Document)) where the first element is the vocab in the corpus and
   *         the second element ae the documents.
   */
  def loadCorpus(start: Int, end: Int) : (Set[String], Seq[Document]) = {
    val docs = (getFolderNames(start, end) map {
      s: String => recursiveListFiles(new File(fileRoot + s))} flatten) map toDocument
    var vocab = Set[String]()
    for (doc: Document <- docs) {
      for (sentence <- doc.sentences) {
        vocab ++= (sentence.tokens map (_.word))
      }
    }
    (vocab, docs)
  }

  def runModel() = {
    // get the training set as a list of documents
    // and train up a unigram model
    print("Loading training and test corpus...")
    val (vocab_training, corpus_training) = loadCorpus(2000, 2005)
    val (vocab_test, corpus_test) = loadCorpus(2005, 2007)
    println("done.")

    println("Training vocab size = %d" format(vocab_training.size))
    println("Test vocab size = %d" format(vocab_test.size))
    println("No of missing words in training vocab = %d" format((vocab_test -- vocab_training).size))


    val constLM = new ConstantLM(vocab_training.size)
    println("ConstantLM perplexity = %.0f" format(perplexity(constLM, corpus_test.iterator)))

    for (order <- 1 until 4) {
      val ngramLM = trainNgramLM(corpus_training, order)
      println("NGram LM order " + order + " perplexity = %.0f" format(perplexity(ngramLM, corpus_test.iterator)))

      val addOneLM = new AddOneLM(ngramLM, vocab_training.size)
      println("NGram LM order " + order + " addOne perplexity = %.0f" format(perplexity(addOneLM, corpus_test.iterator)))
    }
  }

  def main(args: Array[String]) {
    //val vocabulary = loadVocabulary("vocabulary.txt")
    //val history = loadHistory("history.txt")
    //serialize(lm, history, vocabulary, "output.txt")

    runModel
  }
}