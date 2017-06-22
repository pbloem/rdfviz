package nl.peterbloem.rdfviz;

import static nl.peterbloem.kit.Functions.choose;
import static nl.peterbloem.kit.Series.series;

import java.io.BufferedInputStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.nodes.DTGraph;
import org.nodes.DTNode;
import org.nodes.MapDTGraph;
import org.rdfhdt.hdt.exceptions.NotFoundException;
import org.rdfhdt.hdt.hdt.HDT;
import org.rdfhdt.hdt.hdt.HDTManager;
import org.rdfhdt.hdt.triples.IteratorTripleString;
import org.rdfhdt.hdt.triples.TripleString;

import nl.peterbloem.kit.Functions;
import nl.peterbloem.kit.Global;
import nl.peterbloem.kit.Pair;
import nl.peterbloem.kit.Series;

public class HDTIterator implements SentenceIterator 
{
	public static final String TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
	
	// * Big map storing all neighbor relations
	private Map<String, List<Pair<String, String>>> map = 
			new LinkedHashMap<String, List<Pair<String, String>>>();
	
	private List<String> nodes, instances, words;
	
	private Map<String, Integer> wordMap = new HashMap<>();
	
	private int sentenceLength, corpusLength;
	private long seed;
	private Random rand;
	
	private int returned = 0;

	public HDTIterator(File hdtFile, int sentenceLength, int corpusLength, String type, long seed) 
		throws IOException
	{
		Global.log().info("Start loading HDT, filtering by type: " + type);
		
		HDT hdt = HDTManager.loadHDT(
				new BufferedInputStream(new FileInputStream(hdtFile)), null);
		
		Set<String> wordSet = new HashSet<String>();

		int i = 0;
		try {
			// Search pattern: Empty string means "any"
			IteratorTripleString it = hdt.search("", "", "");
						
			while(it.hasNext()) 
			{
				TripleString ts = it.next();

				String subject = ts.getSubject().toString(), 
				       predicate = ts.getPredicate().toString(),
				       object = ts.getObject().toString();
				
				wordSet.add(subject);
				wordSet.add(predicate);
				wordSet.add(object);
				
				// * Add forward
				if(! map.containsKey(subject))
					map.put(subject, new ArrayList<>());
				
				map.get(subject).add(Pair.p(predicate, object));
					
				// * Add backward
				if(! map.containsKey(object))
					map.put(object, new ArrayList<>());

				map.get(object).add(Pair.p(predicate, subject));
				
				Functions.dot(i++, (int)it.estimatedNumResults());
			}
			
			this.nodes = new ArrayList<String>(map.keySet());
			
			if(type == null)
			{
				instances = nodes;
			} else {
				
				Set<String> instSet = new HashSet<String>();
				
				it = hdt.search("", TYPE, type);
				
				while(it.hasNext())
				{
					TripleString ts = it.next();
					String subject = ts.getSubject().toString();
					
					instSet.add(subject);
				}
				
				instances = new ArrayList<String>(instSet);
				
				Global.log().info(instances.size() + " instances extracted.");
			}
		} catch (NotFoundException e) 
		{
			throw new RuntimeException(e);
		} finally 
		{
			// IMPORTANT: Free resources
			hdt.close();
		}
		
		this.words = new ArrayList<>(wordSet);
		for(int j : Series.series(words.size()))
			wordMap.put(words.get(j), j);
		
		Global.log().info("HDT loaded");	
				
		this.sentenceLength = sentenceLength;
		this.corpusLength = corpusLength;
		this.seed = seed;
		this.rand = new Random(seed);
	}

	public String nextSentence() 
	{
		StringBuffer sentence = new StringBuffer();
		
		// * Pick a random node
		String node = choose(instances);
		
		sentence.append(wordMap.get(node) + " ");

		for(int i : Series.series(sentenceLength - 1))
		{		
			Pair<String, String> pair = choose(map.get(node), rand);
			
			sentence.append(wordMap.get(pair.first()) + " ");
			sentence.append(wordMap.get(pair.second()) + " ");
			
			node = pair.second();
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
		map = null;
		nodes = null;
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
	
	/**
	 * Write a CSV file of all instances: their IDs, and their IRIs
	 * 
	 * @param file
	 * @throws IOException 
	 */
	public void saveNames(File file) 
			throws IOException
	{		
		Writer out = new BufferedWriter(new FileWriter(file));
		
		for(String instance : instances)
			out.write(wordMap.get(instance) + ", " + instance + "\n"); 
		
		out.close();
	}

}
