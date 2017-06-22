package nl.peterbloem.rdfviz;

import static nl.peterbloem.kit.Functions.tic;
import static nl.peterbloem.kit.Functions.timeString;
import static nl.peterbloem.kit.Functions.toc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.apache.commons.compress.compressors.FileNameUtil;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nodes.DTGraph;
import org.nodes.UGraph;
import org.nodes.data.RDF;
import org.openrdf.rio.RDFFormat;
import org.rdfhdt.hdt.hdt.HDT;
import org.rdfhdt.hdt.hdt.HDTManager;
import org.rdfhdt.hdt.triples.IteratorTripleString;
import org.rdfhdt.hdt.triples.TripleString;

import nl.peterbloem.kit.FileIO;
import nl.peterbloem.kit.Functions;
import nl.peterbloem.kit.Global;

public class Run {
	
	@Option(name="--file", usage="Input file: HDT format.", required=true)
	private static File file;
	
	@Option(name="--sentence-length", usage="Sentence length.")
	private static int sentenceLength = 7;
	
	@Option(name="--corpus-length", usage="Number of sentences to generate.")
	private static int corpusLength = 5000000;	
	
	@Option(name="--layer-size", usage="The size of the hidden layer.")
	private static int layerSize = 500;	
	
	@Option(name="--window-size", usage="The size of the context that word2vec should reproduce.")
	private static int windowSize = 5;	
	
	@Option(name="--tsne", usage="P roduce 2D vectors with tSNE (slow).")
	private static boolean tsne = false;
	
	@Option(name="--type", usage="Focus the random walks on nodes of the given type.")
	private static String type = null;
	
	public static void main(String[] args) 
			throws IOException
	{
		Run run = new Run();
		
		// * Parse the command-line arguments
    	CmdLineParser parser = new CmdLineParser(run);
    	try
		{
			parser.parseArgument(args);
		} catch (CmdLineException e)
		{
	    	System.err.println(e.getMessage());
	        System.err.println("java -jar motive.jar [options...]");
	        parser.printUsage(System.err);
	        
	        System.exit(1);	
	    }
				
		HDTIterator it = new HDTIterator(file, sentenceLength, corpusLength, type, 0);
		
		TokenizerFactory t = new DefaultTokenizerFactory();
		
        Global.log().info("First five sentences....");
        int i = 0;
        while(it.hasNext() & i++ < 6)
        {
        	String sentence = it.nextSentence();
        	System.out.println("     full sentence: " + sentence);

        	System.out.println("tokenized sentence: ");
        	Tokenizer tokenizer = t.create(sentence);

        	// iterate over the tokens
  	      	while(tokenizer.hasMoreTokens()) {
  	      	   String token = tokenizer.nextToken();
  	      	   System.out.print(token + " ");
  	      	}
  	      	
  	      	System.out.println();
        	
        }
        it.reset();
		
        Global.log().info("Saving names....");

		it.saveNames(new File("names.csv"));
        Global.log().info("Names saved....");
		

		
        Global.log().info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(layerSize)
                .seed(42)
                .windowSize(windowSize)
                .iterate(it)
                .tokenizerFactory(t)
                .build();

        Global.log().info("Fitting Word2Vec model....");
        tic();
        vec.fit();
        Global.log().info("Model finished, time taken: " + timeString(toc()));
         
        int numWords = vec.getVocab().numWords();
        
        WordVectorSerializer.writeWordVectors(vec, new File("full-vectors.tsv"));
        
        if(tsne)
        {
            Global.log().info("Plot TSNE....");
            BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                    .setMaxIter(1000)
                    .stopLyingIteration(250)
                    .learningRate(500)
                    .useAdaGrad(false)
                    .theta(0.5)
                    .setMomentum(0.5)
                    .normalize(true)
                    .build();
            
            vec.lookupTable().plotVocab(tsne, numWords, new File("tsne-vectors.tsv"));
        }
        
        try
		{
			FileIO.python(new File("."), "scripts/plot.py");
		} catch (InterruptedException e)
		{
			Global.log().warning("Failed to run plot script.");
		}
	}

}
