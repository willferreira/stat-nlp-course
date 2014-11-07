package uk.ac.ucl.cs.mr.assignment1

import java.io.{File}
import scala.util.control.Breaks._
import scala.language.postfixOps
import scala.collection.immutable.Range

import ml.wolfe.nlp.{Document, Sentence}

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
   * Calculates the probability of the given sentence using the
   * given language model.
   * @param lm the language model
   * @param sentence the sentence
   */
  def calcSentenceProbability(lm: LanguageModel, sentence: Sentence) : Double = {
    val words = Seq.concat(
      Seq.fill(lm.order - 1) {
        "<s>"
      },
      sentence.tokens map (_.word),
      Seq.fill(lm.order - 1) {
        "</s>"
      }
    )

    val prob = (Range(lm.order - 1, words.length)
      map {i => lm.prob(words(i), words.slice(i - lm.order + 1, i))}).foldLeft(1.0)(_ * _)
    prob
  }

  /**
   * Returns the most likely sentence of length n from the given corpus. If there are no
   * sentence of length n, then find sentences of length m > n.
   * @param documents the corpus of documents
   * @param lm the language model
   * @param n the sentence length
   * @return the most likely sentence of length n
   */
  def calcMaxProbabilityOfSentenceLengthN(documents: Seq[Document], lm: LanguageModel, n: Int) : (Int, (Sentence, Double))  = {
    var sentencesAndLengths: Map[Sentence, Int] = Map()
    for (document <- documents) {
      for (sentence <- document.sentences) {
        sentencesAndLengths += sentence -> sentence.tokens.length
      }
    }

    val reverse = sentencesAndLengths.groupBy({case (_, b) => b})
    val maxLength = reverse.keySet.max
    for (i <- Range(n, maxLength)) {
      if (reverse contains i) {
        println("Calculating most likely sentence for sentences of length %d".format(i))
        println("There are %d sentences of length %d".format(reverse(i).size, i))

        val sentencesOfLengthi = reverse(i).map(_._1)
        val probs = (sentencesOfLengthi map (s => (s, calcSentenceProbability(lm, s))))
        return (i, probs.zipWithIndex.maxBy({case ((s, p), i) => p})._1)
      }
    }
    return (0, null)
  }

  /**
   * Calculate the perplexity of a language model, given a test set
   * of documents. This function
   * will add n-1 start <s> and end </s> of sentence markers to
   * each sentence for a LM of order n, to provide
   * context for the probability estimate.
   * @param lm a LM
   * @param documents an iterator over a collection of Documents
   * @return a Double representing the perplexity measure
   */
  def perplexity(lm: LanguageModel, documents: Iterator[Document]): Double = {
    var s: Double = 0.0
    var N: Double = 0.0

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

        // Remove the number of start-of-sentence
        // markers from the word count N in the perplexity measure to ensure
        // a proper probability distribution
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

  /**
   * A Constant LM.
   * @param vocabSize the training vocab size.
   */
  class ConstantLM(vocabSize: Int) extends LanguageModel {
    val order = 1

    def prob(word: String, history: NGram) = 1.0 / vocabSize
  }

  /**
   * A basic NGram LM with no smoothing.
   * @param countsN a Map of NGram counts
   * @param countsNMinus1 a Map of NGram-1 counts
   * @param order the order of the LM.
   */
  class NgramLM(val countsN: Counts,
                val countsNMinus1: Counts,
                val order: Int) extends LanguageModel {
    def prob(word: String, history: NGram) = {
      (countsN getOrElse(history :+ word, 0.0)) / (countsNMinus1 getOrElse(history, 0.0))
    }
  }

  /**
   * Implements an Add-k smoothed NGram LM.
   * @param ngramLM the LM
   * @param vocab the training vocab
   * @param k the smoothing adjustment
   */
  class AddkLM(ngramLM: NgramLM, vocab: Map[String, Int], k: Double = 1.0, useUnk: Boolean = false) extends LanguageModel {
    def order = ngramLM.order

    def prob(word: String, history: NGram) = {
      // if the word or any in it's history is not in the training vocab, then substitute <UNK>
      val word1 = if (useUnk) if (!(vocab contains word)) "<UNK>" else word else word
      val history1 = if (useUnk) history map {s => if (vocab contains s) s else "<UNK>"} else history
      (k + (ngramLM.countsN getOrElse(history1 :+ word1, 0.0))) / ((k * vocab.size) + (ngramLM.countsNMinus1 getOrElse(history1, 0.0)))
    }
  }

  /**
   * Implements Stupid back-off smoothed NGram LM. This LM employs a naieve form of back-off
   * to smooth the given NGram LM according to the following rules:
   *
   * if order == 0 then return P(word|history) = 1 / vocabSize
   *
   * if order > 0 then
   *  find P(word|history) at order,
   *  if P(word|history) at order is 0 then
   *    find 0.4 * P(word|history) at order - 1, recursively
   *  else
   *    return P(word|history) at order
   * @param ngramLM the given LM
   * @param vocabSize the training vocab size
   */
  class StupidBackoffLM(ngramLM: NgramLM, vocabSize: Int) extends LanguageModel {
    def order = ngramLM.order

    // cache the count maps at each order for later use
    var countMap: Map[Int, (Counts, Counts)] = Map(
      order -> (ngramLM.countsN, ngramLM.countsNMinus1))

    def getCounts(counts: Counts, ord: Int) : (Counts, Counts) = {
      // get counts for the next level down and cache if not
      // yet calculated
      if (!(countMap.contains(ord))) {
        val x: Counts = getNMinus1Counts(counts)
        countMap = countMap updated(ord, (counts, x))
        (counts, x)
      } else {
        countMap(ord)
      }
    }

    // Stupid back-off coefficient
    val alpha = 0.4

    def prob(word: String, history: NGram, countsN: Counts, countNMinus1: Counts, ord: Int) : Double = {
      // try and get a probability from the LM, and if probability is 0 or undefined then back-off
      if (ord == 0)
        // base case is constant LM
        1.0 / vocabSize
      else {
        // try and get a probability at order ord
        val p = (countsN getOrElse(history :+ word, 0.0)) / (countNMinus1 getOrElse(history, 0.0))
        if (p == 0.0 || p.isInfinite || p.isNaN) {
          // back-off
          val (x, y) = getCounts(countNMinus1, ord-1)
          alpha * prob(word, history.drop(1), x, y, ord-1)
        }
        else
          p
      }
    }

    def prob(word: String, history: NGram) = prob(word, history, ngramLM.countsN, ngramLM.countsNMinus1, order)
  }

  /**
   * Trains an NGram LM of the given order in the given document corpus
   * @param train the training set
   * @param order the order of the NGram model, e.g. 1 = unigram, 2 = bigram etc
   * @return an NGram LM
   */
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
    // files before 2000 need to be decoded using UTF-16
    // files on or after 2000 need to be decoded using ISO8859-1
    // nice one!
    Range(start, end) map {i: Int => "P" + (i.toString().takeRight(2))}
  }

  /**
   * Load a corpus of data from the ACL Corpus, given the start and end year.
   * @param start the start year
   * @param end the end year
   * @param addUnk if true then replace the first occurrence of a each word in the documents
   *               by the unknown word marker <UNK>
   * @return a tuple (Map[String, Int], Seq(Document)) where the first element is a multiset
   *         containing the vocab in the corpus, and the second element ae the documents.
   */
  def loadCorpus(start: Int, end: Int, addUnk:Boolean = false) : (Map[String, Int], Seq[Document]) = {

    val docs = (getFolderNames(start, end) map {
      s => recursiveListFiles(new File(fileRoot + s))} flatten) map toDocument
    var vocab = Map[String, Int]()
    for (doc: Document <- docs) {
      for (sentence <- doc.sentences) {
        vocab = vocab ++ (sentence.tokens map {case t => (t.word, 1 + vocab.getOrElse(t.word, 0))})
      }
    }
    if (addUnk)
      addUnks(vocab, docs)
    else
      (vocab, docs)
  }

  /**
   * Runs the AddkLM for a given set of parameters and displays the results.
   *
   * @param k the value to add to smooth, defaults to 1.0
   * @param useUnk whether to use <UNK> for unknown words, defaults to false
   * @param maxOrder the max LM order size to run the model for, defaults to 3.
   */
  def runAddkModel(k: Seq[Double] = Seq(1.0), useUnk: Boolean = false, maxOrder: Int = 3) = {
    val (vocab_training, corpus_training, _, corpus_test) = loadData(useUnk=useUnk)

    val constLM = new ConstantLM(vocab_training.size)
    println("ConstantLM perplexity = %.0f" format(perplexity(constLM, corpus_test.iterator)))

    for (order <- 1 until maxOrder + 1) {
      val ngramLM = trainNgramLM(corpus_training, order)

      for (e <- k) {
        val addkLM = new AddkLM(ngramLM, vocab_training, e, useUnk = useUnk)
        println("NGram LM order %d with add-k smoothing (k=%.4f) has perplexity = %.0f" format(order, e, perplexity(addkLM, corpus_test.iterator)))
      }
    }
  }

  // the location of the ACL Corpus data on my local disk
  var fileRoot = "/Users/williamferreira/Documents/UCL/CSML/Statistical NLP/Assignments/ACL/P/"

  /**
   * Runs the stupid back-off model and displays the results.
   * @param maxOrder the max LM order size to run the model for, defaults to 3.
   */
  def runBackOff(maxOrder: Int = 3) = {
    val (vocab_training, corpus_training, _, corpus_test) = loadData()

    for (order <- 1 until maxOrder + 1) {
      val backoffLM = new StupidBackoffLM(trainNgramLM(corpus_training, order), vocab_training.size)
      println("Stupid back-off LM order %d has perplexity = %.0f" format (order, perplexity(backoffLM, corpus_test.iterator)))
    }
  }

  def runMaxProbableSentence(lm: LanguageModel, documents: Seq[Document], n: Int) = {

  }

  /**
   *
   * @param trainingStart the start year of the training set
   * @param trainingEnd the end year of the training set
   * @param testStart the start year of the test set
   * @param testEnd the end year of the test set
   * @param useUnk whether to use <UNK> for unknown words, defaults to false
   * @return
   */
  def loadData(trainingStart: Int = 1990, trainingEnd: Int = 2005,
               testStart:Int = 2005, testEnd: Int = 2007, useUnk: Boolean = false) = {
    // get the training set as a list of documents
    // and train up some models
    println("Training period: %d to %d" format(trainingStart, trainingEnd-1))
    println("Test period: %d to %d" format(testStart, testEnd-1))

    println("Loading training and test corpus...")
    val (vocab_training, corpus_training) = loadCorpus(trainingStart, trainingEnd, addUnk=useUnk)
    val (vocab_test, corpus_test) = loadCorpus(testStart, testEnd)
    println("done loading corpus data.")

    println("Training vocab size = %d" format vocab_training.size)
    println("Test vocab size = %d" format vocab_test.size)
    println("No. of test data words missing from training vocab = %d" format((vocab_test.keySet -- vocab_training.keySet).size))

    println("Training corpus size = %d" format (vocab_training.foldLeft(0)(_+_._2)))
    println("Test corpus size = %d" format (vocab_test.foldLeft(0)(_+_._2)))

    (vocab_training, corpus_training, vocab_test, corpus_test)
  }

  /**
   * Main method. Can be used to run different scenarios on LMs by
   * supplying the correct command-line parameter.
   * @param args
   */
  def main(args: Array[String]) {
    if (args.length < 3 && args.length > 0) {
      val command = args(0)
      if (args.length == 2)
        fileRoot = args(1)
      command match {
        case "RunAddk" => {
          runAddkModel(k=Seq(2.0, 1.0, 0.1, 0.01, 0.001))
        }

        case "RunAddkWithUnk" => {

        }

        case "RunBackOff" => {
          runBackOff()
        }

        case "CalcMaxSentence" => {
          val (vocab_training, corpus_training, vocab_test, corpus_test) = loadData(trainingStart=1979)
          val ngramLM = trainNgramLM(corpus_training, 2)
          val addkLM = new AddkLM(ngramLM, vocab_training)
          val stupidLM = new StupidBackoffLM(ngramLM, vocab_training.size)
          val (length, (sentence, prob)) = calcMaxProbabilityOfSentenceLengthN(corpus_test, stupidLM, 10)
          println("Most likely sentence of length %d is %s with probability %E"
            .format(length, sentence.tokens map(_.word), prob))
        }

        case _ => println("Unknown case")
      }
    }
  }
}