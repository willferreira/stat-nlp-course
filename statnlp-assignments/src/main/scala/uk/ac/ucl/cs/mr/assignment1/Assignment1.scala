package uk.ac.ucl.cs.mr.assignment1

import java.io.{File}
import scala.language.postfixOps
import scala.collection.immutable.Range

import ml.wolfe.nlp.{Document}

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
   * Calculate the perplexity of a language model, given a test set of documents. This function
   * will add n-1 start <s> and end </s> of sentence markers to each sentence for a LM of order n, to provide
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
   * Implements an Add one smoothed NGram LM.
   * @param ngramLM the LM
   * @param vocab the training vocab
   * @param eps the smoothing adjustment
   */
  class AddOneLM(ngramLM: NgramLM, vocab: Map[String, Int], eps: Double = 1.0, useUnk: Boolean = false) extends LanguageModel {
    def order = ngramLM.order

    def prob(word: String, history: NGram) = {
      // if the word or any in it's history is not in the training vocab, then substitute <UNK>
      val word1 = if (useUnk) if (!(vocab contains word)) "<UNK>" else word else word
      val history1 = if (useUnk) history map {s => if (vocab contains s) s else "<UNK>"} else history
      (eps + (ngramLM.countsN getOrElse(history1 :+ word1, 0.0))) / ((eps * vocab.size) + (ngramLM.countsNMinus1 getOrElse(history1, 0.0)))
    }
  }

  /**
   * Implements Back-off smoothed NGram LM. This LM employs a naieve form of back-off
   * to smooth the given NGram LM according to the following rules:
   *
   * if order == 0 then return P(word|history) = 1 / vocabSize
   *
   * if order > 0 then
   *  find P(word|history) at order,
   *  if P(word|history) at order is 0 then
   *    find P(word|history) at order - 1, recursively
   *  else
   *    return P(word|history) at order
   * @param ngramLM the given LM
   * @param vocabSize the training vocab size
   * @param alpha the back-off coefficient. defaults to 1.0
   */
  class BackoffLM(ngramLM: NgramLM, vocabSize: Int, alpha: Double = 1.0) extends LanguageModel {
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
   * A Simple Good-Turing smoothed NGram LM.
   * @param ngramLM The NGram LM to smooth
   * @param vocabSize the size of the training vocab for the given LM.
   */
  class SimpleGoodTuringLM(ngramLM: NgramLM, vocabSize: Int) extends LanguageModel {
    def order = ngramLM.order

    // data is a Map(c -> Nc)
    val data = ngramLM.countsN.groupBy({case (_, i) => i}) map {case (k, v) => (k -> v.size.toDouble)}
    val (slope, intercept) = estimateLogLinearLSCoefficients(data)

    val N = data.keySet.foldLeft(0.0)({case (a, b) => a + b})

    def getCountStar() = {
      0.0
    }

    def prob(word: String, history: NGram) = {
      0.0
    }
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

  // the location of the ACL Corpus data on my local disk
  val fileRoot = "/Users/williamferreira/Documents/UCL/CSML/Statistical NLP/Assignments/ACL/P/"

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
   * Runs the model(s) for the given range of training and test corpus. Displays results on
   * stdout. NB. no checking is performed on the training or test year ranges.
   * @param trainingStart training start year
   * @param trainingEnd training end tear
   * @param testStart test start year
   * @param testEnd test end year
   */
  def runModel(trainingStart: Int = 1990, trainingEnd: Int = 2005,
                testStart:Int = 2005, testEnd: Int = 2007) = {
    // get the training set as a list of documents
    // and train up some models
    println("Training period: %d to %d" format(trainingStart, trainingEnd-1))
    println("Test period: %d to %d" format(testStart, testEnd-1))

    println("Loading training and test corpus...")
    val useUnk = false
    val (vocab_training, corpus_training) = loadCorpus(trainingStart, trainingEnd, addUnk=useUnk)
    val (vocab_test, corpus_test) = loadCorpus(testStart, testEnd)
    println("done loading corpus data.")

    println("Training vocab size = %d" format(vocab_training.size))
    println("Test vocab size = %d" format(vocab_test.size))
    println("No. of test data words missing from training vocab = %d" format((vocab_test.keySet -- vocab_training.keySet).size))


    val constLM = new ConstantLM(vocab_training.size)
    println("ConstantLM perplexity = %.0f" format(perplexity(constLM, corpus_test.iterator)))

    val eps = 0.01
    for (order <- 1 until 4) {
      val ngramLM = trainNgramLM(corpus_training, order)
//      println("NGram LM order " + order + " has perplexity = %.0f" format(perplexity(ngramLM, corpus_test.iterator)))

      val addOneLM = new AddOneLM(ngramLM, vocab_training, eps, useUnk = useUnk)
      println("NGram LM order " + order + " with add-one smoothing has perplexity = %.0f" format(perplexity(addOneLM, corpus_test.iterator)))

//      val backoffLM = new BackoffLM(ngramLM, vocab_training.size)
//      println("Backoff LM order " + order + " has perplexity = %.0f" format(perplexity(backoffLM, corpus_test.iterator)))
    }
  }

  def main(args: Array[String]) {
    //val vocabulary = loadVocabulary("vocabulary.txt")
    //val history = loadHistory("history.txt")
    //serialize(lm, history, vocabulary, "output.txt")

    runModel()
  }
}