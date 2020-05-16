package lab1.io;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class WekaReader {
    private String wekaFilePath;

    public WekaReader(String wekaFilePath) {
        this.wekaFilePath = wekaFilePath;
    }

    public Instances getData() throws Exception {
        return new ConverterUtils.DataSource(wekaFilePath).getDataSet();
    }
}
