package lab2;

import java.util.*;
import java.util.stream.Collectors;

class CustomWekaImpl {

    private List<ClassificationItem> dataset;

    private double logBase;

    CustomWekaImpl(List<ClassificationItem> dataset, double logBase) {
        this.dataset = dataset;
        this.logBase = logBase;
    }

    double gainRatioAttributeEval(String className) {
        System.out.println("AAAA");
        double entrAttr = countEntropyAttribute();
        if (entrAttr == 0) return 0;
        double entrClass = countEntropyClass(className);
        double entrCond = countEntropyConditional();

        System.out.println("H(CLASS):");
        System.out.println(entrClass);
        System.out.println("H(Attr):");
        System.out.println(entrAttr);
        System.out.println("H(Class|Attr)");
        System.out.println(entrCond);

        return (entrClass - entrCond) / entrAttr;
    }

    List<Double> getMembersOfClass(String className) {
        return dataset.stream()
                .filter(item -> className.equals(item.getGivenClass()))
                .mapToDouble(ClassificationItem::getValue)
                .boxed()
                .collect(Collectors.toList());
    }

    List<Double> getValuesOfAttribute() {
        return dataset.stream()
                .map(ClassificationItem::getValue)
                .collect(Collectors.toList());
    }

    Map<Double, List<String>> getBatches() {
        List<Double> keys = dataset.stream().mapToDouble(ClassificationItem::getValue).distinct().boxed().collect(Collectors.toList());
        Map<Double, List<String>> map = new HashMap<>();
        keys.forEach(key -> {
            List<String> vals = dataset.stream().filter(item -> key.equals(item.getValue())).map(ClassificationItem::getGivenClass).collect(Collectors.toList());
            map.put(key, vals);
        });
        return map;
    }

    double countEntropyClass(String className) {
        return new EntropyUtils<>(getMembersOfClass(className)).countEntropy();
    }

    double countEntropyAttribute() {
        return new EntropyUtils<>(getValuesOfAttribute()).countEntropy();
    }

    // https://stackoverflow.com/questions/33982943/how-the-selection-happens-in-infogainattributeeval-in-weka-feature-selection
    double countEntropyConditional() {
        int size = dataset.size();
        Map<Double, List<String>> batches = getBatches();
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

        EntropyUtils(double logBase) {
            this(new ArrayList<>(), logBase);
        }

        EntropyUtils(List<T> values) {
            this(values, Math.E);
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
            return probab * logn(probab, logBase);
        }

        private double getProbability(T value) {
            int size = values.size();
            if (size == 0) return 0;
            int valueOccurences = (int) (values.stream().filter(d -> d.equals(value)).count());
            return (double)valueOccurences/size;
        }
        private static double logn( double a, double n ) {
            return Math.log(a) / Math.log(n);
        }
    }
}
