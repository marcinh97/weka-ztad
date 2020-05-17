package lab2;

import java.util.*;
import java.util.stream.Collectors;

class CustomWekaImpl<T> {

    private List<ClassificationItem<T>> dataset;

    private double logBase;

    CustomWekaImpl(List<ClassificationItem<T>> dataset, double logBase) {
        this.dataset = dataset;
        this.logBase = logBase;
    }

    double gainRatioAttributeEval() {
        double entrAttr = countEntropyAttribute();
        if (entrAttr == 0) return 0;
        double entrClass = countEntropyClass();
        double entrCond = countEntropyConditional();
        return (entrClass - entrCond) / entrAttr;
    }


    private List<T> getValuesOfAttribute() {
        return dataset.stream()
                .map(ClassificationItem::getValue)
                .collect(Collectors.toList());
    }

    private List<String> getClassesOfAttribute() {
        return dataset.stream()
                .map(ClassificationItem::getGivenClass)
                .collect(Collectors.toList());
    }

    private Map<T, List<String>> getBatches() {
        List<T> keys = dataset.stream().map(ClassificationItem::getValue).distinct().collect(Collectors.toList());
        Map<T, List<String>> map = new HashMap<>();
        keys.forEach(key -> {
            List<String> vals = dataset.stream().filter(item -> key.equals(item.getValue())).map(ClassificationItem::getGivenClass).collect(Collectors.toList());
            map.put(key, vals);
        });
        return map;
    }

    private double countEntropyClass() {
        return new EntropyUtils<>(getClassesOfAttribute(), logBase).countEntropy();
    }

    private double countEntropyAttribute() {
        return new EntropyUtils<>(getValuesOfAttribute(), logBase).countEntropy();
    }

    private double countEntropyConditional() {
        int size = dataset.size();
        Map<T, List<String>> batches = getBatches();
        List<Double> entropies = new ArrayList<>();

        batches.forEach((key, vals) -> {
            double probab = (double)(vals.size()) / size;
            EntropyUtils<String> entropyUtils = new EntropyUtils<>(vals, logBase);
            double result = probab * entropyUtils.countEntropy();
            entropies.add(result);
        });
        return entropies.stream().reduce((d1, d2) -> d1+d2).orElse(0d);
    }

    private static final class EntropyUtils <T> {
        private List<T> values;
        private List<T> valuesDistinct;
        private double logBase;
        private static final double DEFAULT_LOG_BASE = 0.5;

        EntropyUtils(List<T> values) {
            this(values, DEFAULT_LOG_BASE);
        }

        EntropyUtils(List<T> values, double logBase) {
            this.values = values;
            this.logBase = logBase;
            valuesDistinct = values.stream().distinct().collect(Collectors.toList());
        }

        double countEntropy() {
            return (-1) * valuesDistinct.stream().map(this::getSingleEntropy).reduce((d1, d2) -> d1 + d2).orElse(0d);
        }

        private double getSingleEntropy(T value) {
            double probab = getProbability(value);
            return probab * logn(logBase, probab);
        }

        private double getProbability(T value) {
            int size = values.size();
            if (size == 0) return 0;
            int valueOccurrences = (int) (values.stream().filter(d -> d.equals(value)).count());
            return (double)valueOccurrences/size;
        }

        private static double logn(double base, double value){
            return Math.log(value)/Math.log(base);
        }
    }
}
