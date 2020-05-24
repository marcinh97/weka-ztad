package lab3;

import static lab3.Validator.ValidationResult;
import lab1.io.WekaReader;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ResultsAnalyzer {
    private static final String FILES_FOLDER = "src/main/resources/files/";

    private static String toArffFile(String fileName) {
        return FILES_FOLDER + fileName + ".arff";
    }

    private static final double POSITIVE_VALUE_LABEL = 1d;

    private Instances dataset;
    private int testRepetitions;

    public ResultsAnalyzer(Instances dataset, int testRepetitions) {
        this.dataset = dataset;
        this.testRepetitions = testRepetitions;
    }

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

    private ValidationResult testJrip(int folds, String paramName, String param) throws Exception {
        JRip classifier = new JRip();
        classifier.setOptions(new String[] {paramName, param});
        Validator validator = new Validator(dataset, classifier, folds, POSITIVE_VALUE_LABEL);
         return validator.validateClassifier(testRepetitions);
    }

    public static void main(String[] args) {
        WekaReader reader = new WekaReader(toArffFile("238454L3 1"));
        int folds = 10;
        int tests = 10;
        try {
            Instances dataset = reader.getData();
            ResultsAnalyzer resultsAnalyzer = new ResultsAnalyzer(dataset, tests);
            String[] paramsN = new String[]{"0.1", "2", "3", "4", "5", "6", "7", "8", "9", "100"};
            String[] paramsO = {"1", "2", "3", "5", "10", "25"};
            for (String param : paramsN) {
                System.out.println(getSummary(resultsAnalyzer.testJrip(folds, "-N", param), "JRip "+param));
            }
            for (String param : paramsO) {
                System.out.println(getSummary(resultsAnalyzer.testJrip(folds, "-O", param), "JRip "+param));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
