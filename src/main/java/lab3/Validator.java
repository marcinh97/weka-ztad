package lab3;

import lombok.*;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

class Validator {
    private Instances dataset;
    private int folds;
    private Random rand;
    private Classifier classifier;
    private ConfusionMatrix confusionMatrix;

    Validator(Instances dataset, Classifier classifier, int folds, double positiveValueLabel) {
        this.classifier = classifier;
        this.folds = folds;
        int seed = 20;
        this.rand = new Random(seed);
        this.dataset = new Instances(dataset);
        this.confusionMatrix = new ConfusionMatrix(positiveValueLabel);
    }

    ValidationResult validateClassifier(int repetitions) throws Exception {
        for (int repetNum = 0; repetNum<repetitions; repetNum++){
            this.dataset.randomize(rand);
            for (int foldNum = 0; foldNum < folds; foldNum++) {
                Instances train = dataset.trainCV(folds, foldNum, rand);
                Instances test = dataset.testCV(folds, foldNum);
                train.setClassIndex(train.numAttributes() - 1);
                test.setClassIndex(test.numAttributes() - 1);
                classifier.buildClassifier(train);
                testClassifierOnFold(test);
            }
        }
        confusionMatrix.average(repetitions);
        return new ValidationResult(confusionMatrix);
    }

    private void testClassifierOnFold(Instances testSet) throws Exception {
        for (Instance instance : testSet) {
            double realClassVal = instance.classValue();
            double classifierClassVal = classifier.classifyInstance(instance);
            confusionMatrix.update(new SingleValidationResult(realClassVal, classifierClassVal));
        }
    }

    @Data
    @AllArgsConstructor
    static class SingleValidationResult {
        private double realValue;
        private double classifierValue;
    }

    @Getter
    @ToString
    static class ValidationResult {
        private ConfusionMatrix confusionMatrix;
        private double accuracy;
        private double tpRate;
        private double tnRate;
        private double gmean;
        private double auc;

        ValidationResult(ConfusionMatrix confusionMatrix) {
            this.confusionMatrix = confusionMatrix;
            this.accuracy = confusionMatrix.calculateAccuracy();
            this.tpRate = confusionMatrix.calculateTPrate();
            this.tnRate = confusionMatrix.calculateTNrate();
            this.gmean = confusionMatrix.calculateGMean();
            this.auc = confusionMatrix.calculateAUC();
        }
    }
}
