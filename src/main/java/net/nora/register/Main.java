package net.nora.register;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.util.*;
import java.util.stream.Collectors;

import static java.util.Arrays.asList;

public class Main {

    private static Logger log = LoggerFactory.getLogger(Main.class);

    // Constants
    private static final int MIN_VAS_VALUE = 1;
    static final String OUTPUT_HEADER_NAME = "VAS_MK";
    static final int VAS_VALUE_COUNT = 11 - MIN_VAS_VALUE;

    // config values
    static long SEED;
    private static long SHUFFLE_SEED;
    private static int TRAINING_DATA_SIZE;
    static boolean EVALUATE_MODEL_DURING_TRAINING;
    static int EPOCHS;
    private static boolean FILTER_UNKNOWN;
    static boolean FILTER_DUPLICATES;
    private static String DATA_FILE_NAME;
    private static String REGISTER_FILE_NAME;
    static boolean LOAD_ANN_FROM_FILE;
    static String ANN_FILE_NAME;

    // Dependencies
    private static RegisterCsvReader csvReader = new RegisterCsvReader();
    private static RegisterCsvWriter csvWriter = new RegisterCsvWriter();
    private static NetworkDataMapper mapper = new NetworkDataMapper();
    private static NeuronalNetwork deep4jNetwork = new NeuronalNetwork(VAS_VALUE_COUNT);

    public static void main(String[] args) throws Exception {
        String dataDirectoryPath = setupDataDirectory(args);

        readConfig(dataDirectoryPath);

        // prepare training and verification data
        log.info("Reading test data file(" + dataDirectoryPath + "/input/" + DATA_FILE_NAME + ") for training and verification...");
        final RegisterCsvData trainingCsvData = csvReader.read(dataDirectoryPath + "/input/" + DATA_FILE_NAME);
        Collections.shuffle(trainingCsvData.getCsvData(), new Random(SHUFFLE_SEED));
        filterTrainingData(trainingCsvData);
        Map<String, Pair<String, Integer>> answersCaseIdMap = createCaseIdMap(trainingCsvData);
        final RegisterCsvData transformedTrainingData = mapper.transform(trainingCsvData);
        csvWriter.write(transformedTrainingData, TRAINING_DATA_SIZE,
                dataDirectoryPath + "/temp/train_data.csv",
                dataDirectoryPath + "/temp/verification_data.csv");


        // load register data to classify
        log.info("Reading register data file(" + dataDirectoryPath + "/input/" + REGISTER_FILE_NAME + ") to classify with ann...");
        final RegisterCsvData registerCsvData = csvReader.read(dataDirectoryPath + "/input/" + REGISTER_FILE_NAME);
        filterRegisterData(registerCsvData);
        Map<Integer, String> finalRegisterRowNumberToCaseId = createRowToCaseIdMap(registerCsvData);
        final RegisterCsvData transformedRegisterData = mapper.transform(registerCsvData);
        csvWriter.write(transformedRegisterData, dataDirectoryPath + "/temp/register_data.csv");


        int mappedVasIndex = transformedTrainingData.getCsvData().get(0).length - 1;
        int verificationDataSize = transformedTrainingData.getCsvData().size() - TRAINING_DATA_SIZE;
        if (TRAINING_DATA_SIZE < 0) {
            verificationDataSize = transformedTrainingData.getCsvData().size();
        }
        deep4jNetwork.run(
                dataDirectoryPath,
                getTrainingSize(verificationDataSize),
                verificationDataSize,
                mappedVasIndex,
                answersCaseIdMap,
                finalRegisterRowNumberToCaseId);

        System.exit(0);
    }

    private static void readConfig(String dataDirectoryPath) {
        try (InputStream configProperties = new FileInputStream(new File(dataDirectoryPath + "/config.properties"))) {
            Properties prop = new Properties();
            prop.load(configProperties);

            // read the property values
            DATA_FILE_NAME = prop.getProperty("training.data.file");
            SHUFFLE_SEED = Long.parseLong(prop.getProperty("training.data.shuffle.seed"));
            SEED = Long.parseLong(prop.getProperty("ann.initalizing.seed"));
            TRAINING_DATA_SIZE = Integer.parseInt(prop.getProperty("training.data.size"));
            EPOCHS = Integer.parseInt(prop.getProperty("training.epochs"));
            EVALUATE_MODEL_DURING_TRAINING = Boolean.parseBoolean(prop.getProperty("training.evaluate.while.training"));
            REGISTER_FILE_NAME = prop.getProperty("register.data.file");
            FILTER_UNKNOWN = Boolean.parseBoolean(prop.getProperty("training.data.ignore.unknown"));
            FILTER_DUPLICATES = Boolean.parseBoolean(prop.getProperty("training.data.filter.duplicate"));
            LOAD_ANN_FROM_FILE = Boolean.parseBoolean(prop.getProperty("ann.use.saved.ann"));
            ANN_FILE_NAME = prop.getProperty("ann.file.name");

            // log the property values
            System.out.println("===== Configuration =====");
            System.out.println("training.data.file: " + prop.getProperty("training.data.file"));
            System.out.println("training.epochs: " + prop.getProperty("training.epochs"));
            System.out.println("training.evaluate.while.training: " + prop.getProperty("training.evaluate.while.training"));
            System.out.println("training.data.size: " + prop.getProperty("training.data.size"));
            System.out.println("training.data.shuffle.seed: " + prop.getProperty("training.data.shuffle.seed"));
            System.out.println("register.data.file: " + prop.getProperty("register.data.file"));
            System.out.println("ann.initalizing.seed: " + prop.getProperty("ann.initalizing.seed"));
            System.out.println("training.data.ignore.unknown: " + prop.getProperty("training.data.ignore.unknown"));
            System.out.println("training.data.filter.duplicate: " + prop.getProperty("training.data.filter.duplicate"));
            System.out.println("ann.use.saved.ann: " + prop.getProperty("ann.use.saved.ann"));
            System.out.println("ann.file.name: " + prop.getProperty("ann.file.name"));
            System.out.println("===== End Configuration =====");
            System.out.println();

        } catch (IOException e) {
            throw new RuntimeException("Could not read config.properties", e);
        }
    }

    private static String setupDataDirectory(String[] args) throws URISyntaxException {
        if(args.length != 1){
            System.out.println("The path to the data directory needs to be specified as program argument");
            System.exit(1);
        }

        String dataDirectoryPath = args[0];
        if (StringUtils.isEmpty(dataDirectoryPath)) {
            dataDirectoryPath = new File(Main.class.getProtectionDomain().getCodeSource().getLocation().toURI()).getPath() + "/data";
            log.warn("No data directory was set. Using default data directory: " + dataDirectoryPath);
        } else {
            log.info("Using data directory path: " + dataDirectoryPath);
            System.out.println();
            System.out.println();
            System.out.println();
        }
        createMissingDirectories(dataDirectoryPath);
        return dataDirectoryPath;
    }

    private static void createMissingDirectories(String dataDirectoryPath) {
        if (new File(dataDirectoryPath).mkdirs()) {
            log.info("created data directory: " + dataDirectoryPath);
        }
        if (new File(dataDirectoryPath + "/input").mkdirs()) {
            log.info("created input directory: " + dataDirectoryPath + "/input");
        }
        if (new File(dataDirectoryPath + "/temp").mkdirs()) {
            log.info("created temp directory: " + dataDirectoryPath + "/temp");
        }
        if (new File(dataDirectoryPath + "/output").mkdirs()) {
            log.info("created output directory: " + dataDirectoryPath + "/output");
        }
    }

    private static int getTrainingSize(int verificationDataSize) {
        return TRAINING_DATA_SIZE < 0 ? verificationDataSize : TRAINING_DATA_SIZE;
    }

    private static void filterTrainingData(RegisterCsvData csvData) {
        List<String[]> csvLines = csvData.getCsvData();
        int totalCases = csvLines.size();
        csvData.setCsvData(csvLines.stream()
                .filter(csvLine1 -> !shouldRemoveTrainingData(csvLine1))
                .collect(Collectors.toList()));
        int filteredCases = totalCases - csvData.getCsvData().size();
        log.info("Filtered " + filteredCases + " samples of " + totalCases + " total samples.");
    }

    private static void filterRegisterData(RegisterCsvData csvData) {
        List<String[]> csvLines = csvData.getCsvData();
        int totalCases = csvLines.size();
        csvData.setCsvData(csvLines.stream()
                .filter(csvLine1 -> !shouldRemoveRegisterData(csvLine1))
                .collect(Collectors.toList()));
        int filteredCases = totalCases - csvData.getCsvData().size();
        log.info("Filtered " + filteredCases + " row(s) of " + totalCases + " total register rows.");
    }

    private static boolean shouldRemoveTrainingData(String[] csvLine) {
        int vasIndex = csvLine.length - 1;
        if (FILTER_UNKNOWN && asList(csvLine).contains("unknown")) {
            return true;
        }
        if ("".equals(csvLine[vasIndex])) {
            return true;
        }

        int vasValue = Integer.parseInt(csvLine[vasIndex]);
        return vasValue < MIN_VAS_VALUE;
    }

    private static boolean shouldRemoveRegisterData(String[] csvLine) {
        // set some default VAS value that is later overwritten by the ann
        csvLine[csvLine.length - 1] = "0";
        List<String> rowValues = asList(csvLine);
        if (rowValues.contains("unknown")) {
            return true;
        }
        if (rowValues.contains("")) {
            return true;
        }

        return false;
    }

    private static Map<String, Pair<String, Integer>> createCaseIdMap(RegisterCsvData csvData) {
        Map<String, Pair<String, Integer>> lineNumberToCaseId = new HashMap<>();
        List<String[]> csvLines = csvData.getCsvData();
        int vasIndex = csvLines.get(0).length - 1;

        for (int i = 0; i < csvLines.size(); i++) {
            String[] csvLine = csvLines.get(i);
            List<String> answers = new ArrayList<>(asList(csvLine));
            String answersAsKey = toAnswerHashKey(answers);
            Pair<String, Integer> pair = Pair.of(csvLine[0], Integer.parseInt(csvLine[vasIndex]));
            lineNumberToCaseId.put(answersAsKey, pair);
        }

        return lineNumberToCaseId;
    }

    private static Map<Integer, String> createRowToCaseIdMap(RegisterCsvData csvData) {
        Map<Integer, String> createRowToCaseIdMap = new HashMap<>();
        List<String[]> csvLines = csvData.getCsvData();
        for (int i = 0; i < csvLines.size(); i++) {
            String[] csvLine = csvLines.get(i);
            createRowToCaseIdMap.put(i, csvLine[0]);
        }

        return createRowToCaseIdMap;
    }

    public static String toAnswerHashKey(List<String> dataSample) {
        dataSample.remove(dataSample.size() - 1);
        dataSample.remove(0);
        return String.join(",", dataSample);
    }

    public static void printStats(RegisterCsvData transformed, int vasIndex) {
        Map<String, List<Integer>> vasGroups = new HashMap<>();

        for (String[] csvDatum : transformed.getCsvData()) {
            StringBuilder stringBuilder = new StringBuilder();
            for (int i = 0; i < csvDatum.length; i++) {
                if (i != 0 || i != csvDatum.length - 1) {
                    stringBuilder.append(csvDatum[i]);
                }
            }
            String hashedData = stringBuilder.toString();
            List<Integer> vasValues = vasGroups.computeIfAbsent(hashedData, k -> new ArrayList<>());
            vasValues.add(Integer.valueOf(csvDatum[vasIndex]));
        }
        for (Map.Entry<String, List<Integer>> entry : vasGroups.entrySet()) {
            if (entry.getValue().size() > 1) {
                System.out.println(entry.getKey());
                System.out.println(entry.getValue());
                System.out.println();
            }
        }
    }
}
