package lab3;

import static lab3.Validator.ValidationResult;
import lab1.io.WekaReader;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
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

    private ValidationResult testClassifier(AbstractClassifier classifier, int folds, String paramName, String param)
            throws Exception {
        classifier.setOptions(new String[]{paramName, param});
        Validator validator = new Validator(dataset, classifier, folds, POSITIVE_VALUE_LABEL);
        return validator.validateClassifier(testRepetitions);
    }

    private void getSummaryForClassifiers(int folds) throws Exception {
        ValidationResult res = new Validator(dataset, new ZeroR(), folds, POSITIVE_VALUE_LABEL).validateClassifier(testRepetitions);
        System.out.println(getShortSummary(res, "ZeroR"));
        System.out.println();

        String[] params = new String[]{"0.1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20"};
        System.out.println("Parametr -N (minimal weights of instances within a split) ");
        for (String param : params) {
            System.out.println(getShortSummary(testClassifier(new JRip(), folds, "-N", param), "JRip " + param));
        }
        params = new String[]{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20"};
        System.out.println("Parametr -O (number of runs of optimizations) ");
        for (String param : params) {
            System.out.println(getShortSummary(testClassifier(new JRip(), folds, "-O", param), "JRip " + param));
        }

        System.out.println();
        System.out.println("Parametr -C (confidence threshold for pruning) ");
        params = new String[]{"0.01", "0.05", "0.1", "0.25", "0.5"};
        System.out.println("Parametr -C (confidence threshold for pruning) ");
        for (String param : params) {
            System.out.println(getShortSummary(testClassifier(new J48(), folds, "-C", param), "J48 " + param));
        }

        System.out.println();
        params = new String[]{"0.1", "0.2", "0.3", "0.4", "0.5", "1", "2", "3", "4", "5", "10"};
        System.out.println("Parametr -C ");
        for (String param : params) {
            System.out.println(getShortSummary(testSMO(folds, "-C", param), "SMO " + param));
        }

        System.out.println();
        params = new String[]{"0.01", "0.1", "0.2", "0.3", "0.5", "0.75"};
        System.out.println("Parametr -L (learning rate) ");
        for (String param : params) {
            System.out.println(getShortSummary(testClassifier(new MultilayerPerceptron(), folds,
                    "-L", param), "MultilayerPerceptron " + param));
        }

        System.out.println();

        ValidationResult bayesResult = new Validator(dataset, new NaiveBayes(), folds, POSITIVE_VALUE_LABEL).validateClassifier(testRepetitions);
        System.out.println(getShortSummary(bayesResult, "NaiveBayes"));
    }

    private void testForGivenParams(int folds) throws Exception {
        System.out.println();
        System.out.println("Testing all classifiers for " + folds + " folds.");
        ValidationResult res = new Validator(dataset, new ZeroR(), folds, POSITIVE_VALUE_LABEL).validateClassifier(testRepetitions);
        System.out.println(getShortSummary(res, "ZeroR"));

        String param = "1";
        System.out.println(getShortSummary(testClassifier(new JRip(), folds, "-N", param), "JRip "+param));

        param = "0.5";
        System.out.println(getShortSummary(testClassifier(new J48(), folds, "-C", param), "J48 " + param));

        param = "2";
        System.out.println(getShortSummary(testSMO(folds, "-C", param), "SMO " + param));

        param = "0.5";
        System.out.println(getShortSummary(testClassifier(new MultilayerPerceptron(), folds,
                "-L", param), "MultilayerPerceptron " + param));

        ValidationResult bayesResult = new Validator(dataset, new NaiveBayes(), folds, POSITIVE_VALUE_LABEL).validateClassifier(testRepetitions);
        System.out.println(getShortSummary(bayesResult, "NaiveBayes"));
    }

        public static void main(String[] args) {
//        WekaReader reader = new WekaReader(toArffFile("237982L4_1"));
        WekaReader reader = new WekaReader(toArffFile("238454L4 1"));
        int folds = 10;
        int tests = 5;
        int[] foldsArr = {2, 5, 10, 25, 50};
        try {
            Instances dataset = reader.getData();
            ResultsAnalyzer resultsAnalyzer = new ResultsAnalyzer(dataset, tests);
            for (int foldsNum : foldsArr) {
                resultsAnalyzer.testForGivenParams(foldsNum);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
