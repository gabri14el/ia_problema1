package br.uefs.ecomp.ia.sentiment_analysis;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import br.uefs.ecomp.ia.sentiment_analysis.model.Review;

public class InputGenerator {

	private static int total = 5000;
	private static int total_p = (int) (total * (50.0 / 100.0));
	private static int total_trainning = (int) (total * (60.0 / 100.0));
	private static int total_validation = (int) (total * (10.0 / 100.0));
	private static int total_test = (int) (total * (30.0 / 100.0));

	private static List<Review> positives = new LinkedList<>();;
	private static List<Review> negatives = new LinkedList<>();;

	private static Random rand = new Random();

	public static void main(String[] args) throws IOException {
		if (args.length > 1) {
			total = Integer.parseInt(args[0]);

			total_p = Integer.parseInt(args[1]);
			total_p = total * (total_p / 100);

			total_trainning = Integer.parseInt(args[2]);
			total_trainning = total * (total_trainning / 100);

			total_validation = Integer.parseInt(args[3]);
			total_validation = total * (total_validation / 100);

			total_test = 100 - total_trainning - total_validation;
			total_test = total * (total_test / 100);
		}

		readReviews();
		fill();

		List<Review> reviews = new ArrayList<>();
		reviews.addAll(positives);
		reviews.addAll(negatives);
		Comparator<Review> comparator = (r1, r2) -> Integer.compare(r1.getComment().hashCode() * rand.nextInt(), r2.getComment().hashCode() * rand.nextInt());
		Collections.sort(reviews, comparator);

		output(reviews);
	}

	private static void readReviews() throws IOException, UnsupportedEncodingException, FileNotFoundException {
		Review review;
		String[] line;
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(App.COMMENTS_FILE), "UTF-8"))) {
			while (reader.ready()) {
				line = reader.readLine().split(";");
				review = new Review(Integer.parseInt(line[0]), line[1], line[2]);

				if (review.isPositive())
					positives.add(review);
				else if (review.isNegative())
					negatives.add(review);
			}
		}
	}

	private static void fill() {
		if (positives.size() > total_p) {
			positives = positives.subList(0, total_p);
		} else {
			for (int x = 0; positives.size() < total_p; x++)
				positives.add(positives.get(x));
		}

		if (negatives.size() > (total - total_p)) {
			negatives = negatives.subList(0, (total - total_p));
		} else {
			for (int x = 0; negatives.size() < (total - total_p); x++)
				negatives.add(negatives.get(x));
		}
	}

	private static void output(List<Review> reviews) throws IOException {
		int o = 0;
		int n = total_trainning;
		print(reviews.subList(o, n), App.INPUT_TRAINNING_FILE);

		o = n;
		n += total_validation;
		print(reviews.subList(o, n), App.INPUT_VALIDATION_FILE);

		o = n;
		n += total_test;
		print(reviews.subList(o, n), App.INPUT_TEST_FILE);
	}

	private static void print(List<Review> reviews, String fileName) throws UnsupportedEncodingException, FileNotFoundException {
		try (PrintStream print = new PrintStream(new FileOutputStream(fileName), true, "UTF-8")) {
			reviews.forEach((r) -> print.println(r));
		}
	}
}
