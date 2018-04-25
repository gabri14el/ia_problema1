package br.uefs.ecomp.ia.sentiment_analysis.util;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class BagOfWords {

	public static int BINARY = 1;
	public static int TERM_FREQUENCY = 2;

	private List<String> stopWords;
	private List<String> text;
	private List<String> vocabullary;
	private int type;

	public BagOfWords() {
		vocabullary = new ArrayList<>();
		text = new LinkedList<>();
	}

	public void setStopWords(List<String> stopWords) {
		this.stopWords = stopWords;
	}

	public void setType(int type) {
		this.type = type;
	}

	public void addLine(String line) {
		text.add(line);
	}

	public void initialize() {
		for (String t : text) {
			t = clean(t);
			List<String> words = new LinkedList<>(Arrays.asList(t.split("\\s")));
			words.removeAll(stopWords);

			for (String w : words)
				if (!vocabullary.contains(w))
					vocabullary.add(w);
		}
		System.out.println("Tamanho do vocabulário: " + vocabullary.size());
	}

	public int[] createVec(String line) {
		String l = clean(line);
		List<String> words = new LinkedList<>(Arrays.asList(l.split("\\s")));
		words.removeAll(stopWords);

		if (type == BINARY)
			return createBinaryVec(words);
		else if (type == TERM_FREQUENCY)
			return createTFVec(words);

		return new int[vocabullary.size()];
	}

	private int[] createBinaryVec(List<String> words) {
		int[] vec = new int[vocabullary.size()];
		for (String w : words) {
			if (vocabullary.contains(w))
				vec[vocabullary.indexOf(w)] = 1;
		}
		return vec;
	}

	private int[] createTFVec(List<String> words) {
		int[] vec = new int[vocabullary.size()];
		for (String w : words) {
			if (vocabullary.contains(w))
				vec[vocabullary.indexOf(w)]++;
		}
		return vec;
	}

	private String clean(String t) {
		t = Normalizer.normalize(t, Normalizer.Form.NFD);
		t = t.replaceAll("[^\\p{ASCII}]", ""); // Remove qualquer coisa fora da ascii
		t = t.replaceAll("[\\p{InCombiningDiacriticalMarks}]", "");
		t = t.replaceAll("\\d", ""); // Remove números
		t = t.replaceAll("\\s+", " ");
		t = removeDoubleChars(t);
		t = t.trim();

		return t;
	}

	private String removeDoubleChars(String t) {
		t = t.replaceAll("[a]+", "a");
		t = t.replaceAll("[e]+", "e");
		t = t.replaceAll("[i]+", "i");
		t = t.replaceAll("[o]+", "o");
		t = t.replaceAll("[u]+", "u");
		return t;
	}
}
