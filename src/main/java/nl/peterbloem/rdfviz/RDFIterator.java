package nl.peterbloem.rdfviz;

import static nl.peterbloem.kit.Functions.choose;

import java.util.Random;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.nodes.DTGraph;
import org.nodes.DTNode;

import nl.peterbloem.kit.Functions;
import nl.peterbloem.kit.Series;

public class RDFIterator implements SentenceIterator 
{
	private DTGraph<String, String> graph;
	private int sentenceLength, corpusLength;
	private long seed;
	private Random rand;
	
	private int returned = 0;

	public RDFIterator(DTGraph<String, String> graph, int sentenceLength, int corpusLength, long seed) 
	{
		this.graph = graph;
		this.sentenceLength = sentenceLength;
		this.corpusLength = corpusLength;
		this.seed = seed;
		this.rand = new Random(seed);
	}

	public String nextSentence() 
	{
		StringBuffer sentence = new StringBuffer();
		
		// * Pick a random node
		int index = rand.nextInt(graph.size());
		DTNode<String, String> node = graph.get(index);
		
		
		for(int i : Series.series(sentenceLength-1))
		{		
			sentence.append(node.label());

			node = choose(node.neighbors(), rand);
		}
		
		returned ++;
		return sentence.toString();
	}

	public boolean hasNext() 
	{
		return returned < corpusLength;
	}

	public void reset() 
	{
		returned = 0;
		rand = new Random(seed);
	}

	public void finish() 
	{
		graph = null;
	}

	private SentencePreProcessor prep = null;
	
	public SentencePreProcessor getPreProcessor() 
	{
		return prep;
	}

	public void setPreProcessor(SentencePreProcessor preProcessor) 
	{
		prep = preProcessor;
	}

}
