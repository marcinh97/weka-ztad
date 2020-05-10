package io;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaReader {
    private String wekaFilePath;

    public WekaReader(String wekaFilePath) {
        this.wekaFilePath = wekaFilePath;
    }

    public Instances getData() throws Exception {
        return new DataSource(wekaFilePath).getDataSet();
    }

    public String getStructure() throws Exception {
        return new DataSource(wekaFilePath).getRevision();
    }

}
