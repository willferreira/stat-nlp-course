package uk.ac.ucl.cs.mr.assignment1

import java.io.{PrintWriter, File}

import ml.wolfe.nlp.{Document, Sentence, Token, _}
import org.json4s.NoTypeHints
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{read, write}
import uk.ac.ucl.cs.mr.assignment1.Assignment1.LanguageModel
import scala.collection.mutable.Buffer
import scala.io.{Source, Codec}
import scala.collection.immutable.{HashMap, Range}

/**
 * @author Sebastian Riedel
 */
object Assignment1Util {
  /* Type aliases */
  type Counts = Map[NGram,Double]
  type NGram = Seq[String]

  /**
   * Converts a file into a tokenized and sentence segmented document
   * @param file text file to convert
   * @return Document of sentences and tokens.
   */
  def toDocument(file: File): Document = {
    val content = Source.fromFile(file)(Codec("ISO8859-1")).getLines().toArray
    val text = content.foldLeft("")(_ + " " + _)
    val pipeline = SentenceSplitter andThen TokenSplitter
    pipeline(text)
  }

  /**
   * Replace first occurrence of each word in Seq[Document] with the token <UNK>
   * @param docs a Seq[Document]
   * @param vocab a Set[String] containing the vocab in docs
   * @return a tuple (Map[String, Int], Seq[Document]) containing the new vocab abd documents
   */
  def addUnks(vocab: Map[String, Int], docs: Seq[Document]) : (Map[String, Int], Seq[Document]) = {
    println("Performing <UNK> substitutions...")
    var localVocab = scala.collection.mutable.Map[String, Int]() ++ vocab
    var vocabWords = scala.collection.mutable.HashSet[String]() ++ vocab.keySet

    def replaceFirstOccurrenceWithUnk(t: Token) : Token = {
      if (vocabWords contains t.word) {
        // once we replace the first occurrence of a word with <UNK> we
        // don't do that again
        vocabWords -= t.word
        val count: Option[Int] = localVocab.get(t.word)
        count match {
          // decrement the count and remove if we're at zero since we used up the word
          // to create an <UNK>
          case Some(i: Int) => if (i == 1) localVocab -= t.word else localVocab(t.word) = i - 1
          case None =>
        }

        // add <UNK> to the vocab, with its count
        localVocab("<UNK>") = localVocab.getOrElse("<UNK>", 0) + 1
        // replace the actual word with <UNK>
        // NB. the offsets will be wrong, but we don't use them
        Token("<UNK>", t.offsets)
      } else
      // we must have replaced the first occurrence of this word already
        t
    }

    var newDocs: Buffer[Document] = Buffer()
    for (doc <- docs) {
      if (vocabWords.isEmpty)
        newDocs += doc
      else {
        var newSentences: Buffer[Sentence] = Buffer()
        for (sentence <- doc.sentences) {
          if (vocabWords.isEmpty)
          // no need to replace any more word first occurrences
          // as they've all been done
            newSentences += sentence
          else
            newSentences += Sentence(sentence.tokens map replaceFirstOccurrenceWithUnk)
        }
        newDocs += Document(doc.source, newSentences, filename = doc.filename, ir = doc.ir)
      }
    }

    println("done with <UNK> substitutions.")
    (localVocab.toMap, newDocs)
  }

  /**
   * Estimates a log-linear model (used for the Good-Turing language model):
   *  log Nc = a + b log c
   * @param data the data for the model represented as a Map(c -> Nc)
   * @return the (slope, intercept) of the estimated
   */
  def estimateLogLinearLSCoefficients(data: Map[Double, Double]): (Double, Double) = {
    val logData = data.toSeq map {case (c, nc) => (Math.log(c), Math.log(nc))}

    val numerator = (logData map ({case (logc, logNc) => logc * logNc})).sum
    val denominator = (logData map ({case (logc, _) => Math.pow(logc, 2)})).sum
    val slope = numerator / denominator
    val intercept = (logData map ({case (_, logNc) => logNc})).sum / logData.size
    (slope, intercept)
  }

  /**
   * Generic method to get n-grams from a sentence
   * @param sentence sentence to n-grams from
   * @param n n-gram order
   * @return n-grams in sentence
   */
  def ngramsInSentence(sentence: Sentence, n: Int): Seq[NGram] = {
    if (n == 0)
      // 0grams are effectively just a count of the
      // number of words in the sentence, but also
      // add 2 for start and end of sentence markers
      Seq.fill(sentence.tokens.length + 2){Seq()}
    else {
      // prepend/append enough start/end of sentence markers
      // before computing ngrams
      val tokens = Seq.concat(Seq.fill(n) {
        "<s>"
      }, sentence.tokens map (_.word), Seq.fill(n) {
        "</s>"
      })
      Range(0, tokens.length - n + 1) map (i => tokens.slice(i, i + n))
    }
  }

  /**
   * Get all descendant files in the directory specified by f.
   * @param f the parent directory.
   * @return all files in the directory, recursive.
   */
  def recursiveListFiles(f: File): Array[File] = {
    // only interested in .txt files and directories
    val these = f.listFiles.filter(f => (f.getName.endsWith(".txt") || f.isDirectory))
    these ++ these.filter(_.isDirectory).flatMap(recursiveListFiles)
  }

  /**
   * Add counts to an immutable map
   * @param counts map with counts
   * @param ngram ngram to increase count
   * @param count count to add
   * @return new map with count of ngram increased by count.
   */
  def addNgramCount(counts: Counts, ngram: NGram, count: Double = 1.0): Counts = {
    counts updated (ngram, count + (counts getOrElse (ngram, 0.0)))
  }

  /**
   * Take two count maps, and add their counts
   * @param countsMany first count map
   * @param countsFew second count map
   * @return a map that maps each key in the union of keys of countsMany and countsFew to the sum of their counts.
   */
  def addNgramCounts(countsMany: Counts, countsFew: Counts): Counts = {
    countsMany ++ (countsFew map {case (k, v) => (k, countsMany.getOrElse(k, 0.0) + v)})
  }

  /**
   * Collect n-gram counts of a given order in the document
   * @param document the document to get n-gram counts from
   * @param n the n-gram order.
   * @return a n-gram to count in document map
   */
  def getNGramCounts(document: Document, n: Int): Counts = {
      (document.sentences map (
        s => ngramsInSentence(s, n).foldLeft(Map[NGram, Double]())(
          (m, g) => addNgramCount(m, g)))).foldLeft(Map[NGram, Double]())(
            (p, q) => addNgramCounts(p, q))
   }

  /**
   * For a given n-gram count map get the counts for n-1 grams.
   * @param counts the n-gram counts
   * @return the n-1 gram counts.
   */
  def getNMinus1Counts(counts: Counts): Counts = {
    if (counts.isEmpty)
      counts
    else {
      // determine the order using the first element in the map
      val (ngram, _) = counts.head
      val n = ngram.length

      // truncate each ngram to first n-1 elements and grab associated counts
      val nMinus1Counts = counts.foldLeft(Map[NGram, Double]()) {
        case (m, (ngram, count)) => addNgramCount(m, ngram.take(n - 1), count)
      }

      // ensure number of sentence starts is equal to number of ends
      val noSentences = nMinus1Counts(Seq.fill(n - 1) {
        "</s>"
      })
      nMinus1Counts updated(Seq.fill(n - 1) {
        "<s>"
      }, noSentences)
    }
  }

  // loads both the vocabulary and the history file, creates their
  // cartesian product, gets the probability for the language model
  // (account for possibly unknown words!) and saves it in a tab-separated
  // file in the format: historyngram1 historyngram2 word probability
  def serialize(lm: LanguageModel, history: Seq[NGram], vocabulary: Seq[String], filename: String) = {
    val output = new PrintWriter(new File(filename))
    try {
      for {
        ngram <- history
        word <- vocabulary
      } {
        val probability = lm.prob(word, ngram) //what if the word you're searching for is not in your LM?
        output.print("%s\t%s\t%s\t".format(ngram(0), ngram(1), word))
        output.println(probability)
      }
    }
    finally {
      output.close()
    }
  }

  // reads a vocabulary from a file
  // the file is structured as a single vocabulary word per line
  def loadVocabulary(filename: String): Seq[String] = {
    val source = Source.fromFile(filename).getLines()
    source.toSeq
  }

  // reads history (ngrams) from a file
  // the file is structured as a tab-separated pair of words (2-gram) per line
  def loadHistory(filename: String): Seq[NGram] = {
    val source = Source.fromFile(filename).getLines()
    source.map(_.split("\\t").toSeq).toSeq
  }

}








