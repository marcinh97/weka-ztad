package lab1.analyzer;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.SubsetByExpression;

public final class WekaAnalyzer {

    private WekaAnalyzer() { }

    public static Instances filterByExpression(Instances dataset, String customFilter) throws Exception {
        SubsetByExpression filter = new SubsetByExpression();
        filter.setOptions(new String[] { "-E", customFilter});
        filter.setInputFormat(dataset);
        return SubsetByExpression.useFilter(dataset, filter);
    }

    public static Instances removeAttribute(Instances dataset, int attributeIndex) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(new int[] { attributeIndex });
        remove.setInvertSelection(false);
        remove.setInputFormat(dataset);
        return Filter.useFilter(dataset, remove);
    }
}
