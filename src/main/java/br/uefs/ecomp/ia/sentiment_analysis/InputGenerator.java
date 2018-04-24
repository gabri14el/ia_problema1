package br.uefs.ecomp.ia.sentiment_analysis;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import br.uefs.ecomp.ia.sentiment_analysis.model.Review;

public class InputGenerator {

	public static void main(String[] args) throws IOException {
		int p = Integer.parseInt(args[0]);
		int n = Integer.parseInt(args[1]);
		boolean separe = (args.length == 3) ? Boolean.parseBoolean(args[2]) : true;

		List<Review> positives = new LinkedList<>();
		List<Review> negatives = new LinkedList<>();

		Review review;
		String[] line;
		try (BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(App.COMMENTS_FILE), "UTF-8"))) {
			while (reader.ready() && (p > 0 || n > 0)) {
				line = reader.readLine().split(";");
				review = new Review(Integer.parseInt(line[0]), line[1], line[2]);

				if (review.isPositive() && p > 0) {
					positives.add(review);
					p--;
				} else if (review.isNegative() && n > 0) {
					negatives.add(review);
					n--;
				}
			}
		}

		for (int x = 0; p > 0; p--, x++)
			positives.add(positives.get(x));
		for (int x = 0; n > 0; n--, x++)
			negatives.add(negatives.get(x));

		List<Review> reviews = new LinkedList<>();
		reviews.addAll(positives);
		reviews.addAll(negatives);
		if (!separe) {
			Random rand = new Random();
			Collections.sort(reviews, (r1, r2) -> Integer.compare(r1.getComment().hashCode() * rand.nextInt(), r2.getComment().hashCode() * rand.nextInt()));
		}

		try (PrintStream print = new PrintStream(new FileOutputStream(App.INPUT_FILE), true, "UTF-8")) {
			reviews.forEach((r) -> print.println(r));
		}
	}
}
