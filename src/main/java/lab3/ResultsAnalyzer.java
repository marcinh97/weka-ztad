package lab3;

import static lab3.Validator.ValidationResult;
import lab1.io.WekaReader;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
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


    private static String getShortSummary(ValidationResult res, String classificatorName) {
        return String.format("%s: GMean: %.4f, AUC: %.4f", classificatorName, res.getGmean(), res.getAuc());
    }


    private ValidationResult testJrip(int folds, String paramName, String param) throws Exception {
        JRip classifier = new JRip();
        classifier.setOptions(new String[]{paramName, param});
        Validator validator = new Validator(dataset, classifier, folds, POSITIVE_VALUE_LABEL);
        return validator.validateClassifier(testRepetitions);
    }

    private ValidationResult testJ48(int folds, String paramName, String param) throws Exception {
        J48 classifier = new J48();
        classifier.setOptions(new String[]{paramName, param});
        Validator validator = new Validator(dataset, classifier, folds, POSITIVE_VALUE_LABEL);
        return validator.validateClassifier(testRepetitions);
    }

    private ValidationResult testSMO(int folds, String paramName, String param) throws Exception {
        SMO classifier = new SMO();
        classifier.setOptions(new String[]{paramName, param});
        Validator validator = new Validator(dataset, classifier, folds, POSITIVE_VALUE_LABEL);
        return validator.validateClassifier(testRepetitions);
    }

    public static void main(String[] args) {
        WekaReader reader = new WekaReader(toArffFile("237982L4_1"));
        int folds = 10;
        int tests = 10;
        try {
            Instances dataset = reader.getData();
            ResultsAnalyzer resultsAnalyzer = new ResultsAnalyzer(dataset, tests);
            String[] paramsN = new String[]{"0.1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20"};
            String[] paramsO = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20"};
//            System.out.println("Parametr -N (minimal weights of instances within a split) ");
//            for (String param : paramsN) {
//                System.out.println(getShortSummary(resultsAnalyzer.testJrip(folds, "-N", param), "JRip "+param));
//            }
//            System.out.println("Parametr -O (number of runs of optimizations) ");
//            for (String param : paramsO) {
//                System.out.println(getShortSummary(resultsAnalyzer.testJrip(folds, "-O", param), "JRip "+param));
//            }
//            String[] paramsC = {"0.01", "0.05", "0.1", "0.25", "0.5"};
//            System.out.println("Parametr -C (confidence threshold for pruning) ");
//            for (String param : paramsC) {
//                System.out.println(getShortSummary(resultsAnalyzer.testJ48(folds, "-C", param), "J48 " + param));
//            }

            String[] paramsSMO = {"0.00001", "0.0001", "0.001", "0.01", "0.1", "0.2", "0.5"};
            System.out.println("Parametr -C (complexity constant C) ");
            for (String param : paramsSMO) {
                System.out.println(getShortSummary(resultsAnalyzer.testSMO(folds, "-L", param), "SMO " + param));
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
