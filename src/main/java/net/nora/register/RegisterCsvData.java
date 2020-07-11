package net.nora.register;

import java.util.List;

public class RegisterCsvData {

    private List<String[]> csvData;

    private String[] header;

    public List<String[]> getCsvData() {
        return csvData;
    }

    public void setCsvData(List<String[]> csvData) {
        this.csvData = csvData;
    }

    public String[] getHeader() {
        return header;
    }

    public void setHeader(String[] header) {
        this.header = header;
    }
}
