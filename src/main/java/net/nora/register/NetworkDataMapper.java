package net.nora.register;

import java.util.*;

class NetworkDataMapper {

    private static Map<Integer, String> yesNoUnknown = NeuronalNetwork.readEnumCSV("/dictionary/no_yes_unknown_answers.csv");
    private static Map<Integer, String> elicitors = NeuronalNetwork.readEnumCSV("/dictionary/elicitors.csv");
    private static Map<Integer, String> sex = NeuronalNetwork.readEnumCSV("/dictionary/sex.csv");
    private static Map<Integer, String> vas = NeuronalNetwork.readEnumCSV("/dictionary/vas.csv");
    private static Map<String, Integer> yesNoUnknownR = Util.reverseMap(yesNoUnknown);
    private static Map<String, Integer> elicitorsR = Util.reverseMap(elicitors);
    private static Map<String, Integer> sexR = Util.reverseMap(sex);
    private static Map<String, Integer> vasR = Util.reverseMap(vas);

    NetworkDataMapper() {
    }

    void insertData(int[] output, int bitCount, int offest, int value) {
        String binaryValue = Integer.toBinaryString(value);
        if (bitCount < binaryValue.length()) {
            throw new RuntimeException("value: " + value + " does not fit into " + bitCount + " bits");
        }
        // prepend leading zeros
        while (binaryValue.length() < bitCount) {
            binaryValue = "0" + binaryValue;
        }
        for (int i = 0; i < bitCount; i++) {
            output[offest + i] = Integer.parseInt(String.valueOf(binaryValue.charAt(i)));
        }
    }

    private int determineHeaderIndex(String[] header, String outputHeaderName) {
        for (int i = 0; i < header.length; i++) {
            if (header[i].equalsIgnoreCase(outputHeaderName)) {
                return i;
            }
        }
        return -1;
    }

    RegisterCsvData transform(RegisterCsvData csvData) {
        final RegisterCsvData data = new RegisterCsvData();
        List<String[]> transformed = new ArrayList<>();
        data.setCsvData(transformed);

        for (String[] csvLine : csvData.getCsvData()) {
            transformed.add(transform(csvLine, csvData.getHeader()));
        }

        return data;
    }

    private String[] transform(String[] csvLine, String[] header) {
        // remove case_id
        String[] transformed = Arrays.copyOfRange(csvLine, 1, csvLine.length);

        // switch position of VAS value (output) with last column
        int vasIndex = determineHeaderIndex(header, Main.OUTPUT_HEADER_NAME) - 1;
        int lastIndex = transformed.length - 1;
        String vasValue = transformed[vasIndex];
        transformed[vasIndex] = transformed[lastIndex];
        transformed[lastIndex] = vasValue;

        for (int i = 0; i < transformed.length; i++) {
            String value = transformed[i];
            transformed[i] = transformValue(value);
        }

        return transformed;
    }

    private String transformValue(String value) {
        if (yesNoUnknownR.get(value) != null) {
            return yesNoUnknownR.get(value).toString();
        }
        if (sexR.get(value) != null) {
            return sexR.get(value).toString();
        }
        if (elicitorsR.get(value) != null) {
            return elicitorsR.get(value).toString();
        }
        if (vasR.get(value) != null) {
            return vasR.get(value).toString();
        }
        return value;
    }
}
