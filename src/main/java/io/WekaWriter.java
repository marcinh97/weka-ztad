package io;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class WekaWriter {
    private String fileName;

    public WekaWriter(String fileName) {
        this.fileName = fileName;
    }

    public void saveData(Instances dataset) throws Exception {
        ConverterUtils.DataSink.write(fileName, dataset);
    }
}
