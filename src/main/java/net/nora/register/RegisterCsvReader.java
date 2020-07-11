package net.nora.register;

import com.opencsv.*;
import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.util.*;

public class RegisterCsvReader {

    private static final char SEPARATOR = ',';

    private boolean printRecords = false;
    private boolean printVasDistribution = false;

    public RegisterCsvData read(final String filename) {
        List<String[]> csvLines = new ArrayList<>();
        RegisterCsvData result = new RegisterCsvData();
        try (final Reader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(filename)))) {

            final CSVParser parser = new CSVParserBuilder().withSeparator(SEPARATOR).withIgnoreQuotations(true).build();
            final CSVReader csvReader = new CSVReaderBuilder(reader).withSkipLines(0).withCSVParser(parser).build();

            // Reading Records One by One in a String array
            String[] nextRecord;
            String[] header = csvReader.readNext();
            int outputHeaderindex = determineHeaderIndex(header, Main.OUTPUT_HEADER_NAME);

            List<Set<String>> values = new ArrayList<>();
            for (int i = 0; i < header.length; i++) {
                values.add(new HashSet<>());
            }
            while ((nextRecord = csvReader.readNext()) != null) {
                for (int i = 0; i < nextRecord.length; i++) {
                    if (printRecords) {
                        System.out.println(header[i] + ": " + nextRecord[i]);
                    }
                    values.get(i).add(nextRecord[i]);
                }
                csvLines.add(nextRecord);
            }
            if(printRecords){
                printValues(header, values);
            }
            List<String[]> uniqueLines = recordStatistics(csvLines, outputHeaderindex);
            result.setHeader(header);

            if(Main.FILTER_DUPLICATES){
                result.setCsvData(uniqueLines);
            }else{
                result.setCsvData(csvLines);
            }
            return result;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
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

    private List<String[]> recordStatistics(List<String[]> lines, int outputHeaderindex) {
        int unknownLines = 0;
        int noVasValueLines = 0;

        int[] vasDistribution = new int[11];

        Map<String, String> combinations = new HashMap<>();
        Map<String, List<String>> confusions = new HashMap<>();
        List<String[]> uniqueLines = new ArrayList<>();
        int confusionCount = 0;
        int duplicates = 0;

        for (String[] line : lines) {

            // find duplicate inputs, that lead to different outputs that could confuse the network
            String caseId = line[0];
            String vasValue = line[outputHeaderindex];
            List<String> inputValues = new LinkedList<>(Arrays.asList(line));
            inputValues.remove(outputHeaderindex);
            inputValues.remove(0);
            String combinationAsKey = String.join("", inputValues);
            if (combinations.containsKey(combinationAsKey)) {
                duplicates++;
                if (!combinations.get(combinationAsKey).equals(vasValue)) {
                    confusionCount++;
                    confusions.get(combinationAsKey).add(caseId);
                }
            } else {
                combinations.put(combinationAsKey, vasValue);
                confusions.put(combinationAsKey, new LinkedList<>(Collections.singletonList(caseId)));
                uniqueLines.add(line);
            }

            for (int i = 0; i < line.length; i++) {
                if (line[i].equalsIgnoreCase("unknown")) {
                    unknownLines++;
                    break;
                }
            }
            if (StringUtils.isEmpty(vasValue)) {
                noVasValueLines++;
                continue;
            }
            vasDistribution[Integer.valueOf(vasValue)]++;
        }
        System.out.println("total lines: " + lines.size());
        System.out.println("unknown lines: " + unknownLines);
        System.out.println("Missing VAS value lines: " + noVasValueLines);
        System.out.println("Confusions: " + confusionCount);
        System.out.println("Duplicates: " + duplicates);
        System.out.println();
        for (List<String> confusionCaseIds : confusions.values()) {
            if(confusionCaseIds.size() > 1){
                System.out.println("b_case_id == " + String.join(" | b_case_id == ", confusionCaseIds));
            }
        }
        if(printVasDistribution){
            for (int i = 0; i < vasDistribution.length; i++) {
                System.out.println(i + ": " + vasDistribution[i]);
            }
        }
        return uniqueLines;
    }

    private void printValues(String[] header, List<Set<String>> values) {
        for (int i = 0; i < values.size(); i++) {
            Set<String> valueLine = values.get(i);
            if (valueLine != null && valueLine.size() > 0) {
                String headline = header[i];
                System.out.print(i + " ," + headline + ": ");
                System.out.println(String.join(", ", valueLine));
            }
        }
    }
}
