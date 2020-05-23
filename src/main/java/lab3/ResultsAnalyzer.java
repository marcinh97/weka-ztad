package lab3;

import static lab3.Validator.ValidationResult;
import lab1.io.WekaReader;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;

public class ResultsAnalyzer {
    private static final String FILES_FOLDER = "src/main/resources/files/";
    private static String toArffFile(String fileName) {
        return FILES_FOLDER + fileName + ".arff";
    }
    private static final double POSITIVE_VALUE_LABEL = 0d;

    private static String getSummary(ValidationResult validationResult, String classificatorName) {
        return "Results for : " +
                classificatorName +
                "\n" +
                "Matrix: " +
                validationResult.getConfusionMatrix() +
                "\n" +
                "Accuracy: " +
                validationResult.getAccuracy() +
                "\n" +
                "TPrate: " +
                validationResult.getTpRate() +
                "\n" +
                "TNrate: " +
                validationResult.getTnRate() +
                "\n" +
                "GMean: " +
                validationResult.getGmean() +
                "\n" +
                "AUC: " +
                validationResult.getAuc() +
                "\n";
    }



    public static void main(String[] args) {
        WekaReader reader = new WekaReader(toArffFile("238454L3 1"));
        int folds = 10;
        int tests = 20;
        try {
            Instances dataset = reader.getData();
            Classifier classifier = new NaiveBayes();
            Validator validator = new Validator(dataset, classifier, folds, POSITIVE_VALUE_LABEL);
            System.out.println(getSummary(validator.validateClassifier(tests), "Naive Bayes"));

            classifier = new ZeroR();
            validator = new Validator(dataset, classifier, folds, POSITIVE_VALUE_LABEL);
            System.out.println(getSummary(validator.validateClassifier(tests), "Zero Rule"));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
