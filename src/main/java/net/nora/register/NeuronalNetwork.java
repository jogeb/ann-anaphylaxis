package net.nora.register;

import com.opencsv.CSVWriter;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class NeuronalNetwork {

    private static Logger log = LoggerFactory.getLogger(NeuronalNetwork.class);

    private boolean printVasDistribution = false;

    private static Map<Integer, String> yesNo = readEnumCSV("/dictionary/no_yes_answers.csv");
    private static Map<Integer, String> yesNoUnknown = readEnumCSV("/dictionary/no_yes_unknown_answers.csv");
    private static Map<Integer, String> elicitors = readEnumCSV("/dictionary/elicitors.csv");
    private static Map<Integer, String> vas = readEnumCSV("/dictionary/vas.csv");
    private int vasValues;

    public NeuronalNetwork(int vasValues) {
        this.vasValues = vasValues;
    }

    void run(String dataDirectoryPath, int trainingDataSize, int verificationDataSize, int vasIndex, Map<String, Pair<String, Integer>> answersToCaseId, Map<Integer, String> finalRegisterRowNumberToCaseId) {
        log.info("vasIndex: " + vasIndex);
        log.info("trainingDataSize: " + trainingDataSize);
        log.info("verificationDataSize: " + verificationDataSize);
        try {
            //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
            //11 classes (types of vas) in the vas data set. Vas values have integer values 0 to 10

            //VAS training data set: We are loading all of them into one DataSet
            DataSet trainingData = readCSVDataset(
                    dataDirectoryPath + "/temp/train_data.csv",
                    trainingDataSize, vasIndex, vasValues);

            // this is the data we want to classify for verification
            DataSet verificationDataNormalized = readCSVDataset(dataDirectoryPath + "/temp/verification_data.csv",
                    verificationDataSize, vasIndex, vasValues);

            // make the data model for records prior to normalization, because it changes the data.
            Map<Integer, Map<String, Object>> verificationDataRaw = makeRegisterForTesting(verificationDataNormalized);

            // this is the data we want to classify for the first time with the trained nnn
            DataSet registerDataNormalized = readCSVDataset(dataDirectoryPath + "/temp/register_data.csv",
                    7600, vasIndex, vasValues);
            Map<Integer, Map<String, Object>> registerDataRaw = makeRegisterForTesting(registerDataNormalized);

            //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
            normalizer.transform(trainingData);     //Apply normalization to the training data
            normalizer.transform(verificationDataNormalized);         //Apply normalization to the test data. This is using statistics calculated from the *training* set
            normalizer.transform(registerDataNormalized);     //Apply normalization to the register data

            //Configure neural network
            final int numInputs = 24;
            int outputNum = vasValues;
            int epochs = Main.EPOCHS;
            double learningRate = 0.6;
            long seed = Main.SEED;
            log.info("model seed: {}", seed);

            final MultiLayerNetwork model;
            if(Main.LOAD_ANN_FROM_FILE){
                log.info("Skipping training and load the model from file...");
                File file = new File(dataDirectoryPath + "/ann/" + Main.ANN_FILE_NAME);
                model = MultiLayerNetwork.load(file, false);
            }else {
                log.info("Build model....");
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .activation(Activation.TANH)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Sgd(learningRate))
                        .l2(1e-4)
                        .list()
                        .layer(new DenseLayer.Builder()
                                .nIn(numInputs)
                                .nOut(33)
                                .build())
                        .layer(new DenseLayer
                                .Builder()
                                .nOut(33)
                                .build())
                        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .activation(Activation.SOFTMAX)
                                .nOut(outputNum).build())
                        .build();

                //run the model
                model = new MultiLayerNetwork(conf);
                model.init();
                model.setListeners(new ScoreIterationListener(1000));

                log.info("Training model....");
                for (int i = 0; i < epochs; i++) {
                    if (i % 10000 == 0) {
                        if(Main.EVALUATE_MODEL_DURING_TRAINING) {
                            evaluateModel(verificationDataNormalized, model);
                        }
                    }
                    if (i % 25000 == 0) {
                        log.info("setting the learningRate from {} to {}", learningRate, learningRate -= 0.05);
                        model.setLearningRate(learningRate);
                    }

                    model.fit(trainingData);
                }

                log.info("Evaluating model....");
                INDArray evaluationOutput = evaluateModel(verificationDataNormalized, model);

                setFittedClassifiers(evaluationOutput, verificationDataRaw);
                setCaseIdAndActualClassifiert(answersToCaseId, verificationDataRaw);
                logCaseAsCsv(verificationDataRaw, dataDirectoryPath, "verification.csv");
            }

            log.info("Calculating vas for register....");
            INDArray finalRegisterOutput = calculateModel(registerDataNormalized, model);
            setFittedClassifiers(finalRegisterOutput, registerDataRaw);
            setCaseIdOfFinalRegisterData(finalRegisterRowNumberToCaseId, registerDataRaw);
            logCaseAsCsv(registerDataRaw, dataDirectoryPath, "ann_calculated.csv");
            log.info("Wrote ann calculated vas values as column 'vas_score_ann' to: "
                    + dataDirectoryPath + "/output/" + "ann_calculated.csv");

            int[] vasDistribution = new int[11];
            for (Map<String, Object> outputLine : registerDataRaw.values()) {
                int vas_score_ann = Integer.parseInt(outputLine.get("vas_score_ann").toString());
                vasDistribution[vas_score_ann] = vasDistribution[vas_score_ann] + 1;
            }

            if(printVasDistribution){
                for (int i = 0; i < vasDistribution.length; i++) {
                    System.out.println(i + ": " + vasDistribution[i]);
                }
            }

            File file = new File(dataDirectoryPath + "/output/model.nnet");
            model.save(file);

        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private INDArray evaluateModel(DataSet testData, MultiLayerNetwork model) {
        Evaluation eval = new Evaluation(Main.VAS_VALUE_COUNT);
        INDArray output = model.output(testData.getFeatures());

        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());
        return output;
    }

    private INDArray calculateModel(DataSet registerData, MultiLayerNetwork model) {
        return model.output(registerData.getFeatures());
    }

    static void logCaseAsCsv(Map<Integer, Map<String, Object>> cases, String dataDirectoryPath, String fileName) {
        String[] header = new String[]{"case_id",
                "skin",
                "q_111_angioedema",
                "pharynx_larynx",
                "abdomin",
                "nausea",
                "vomiting",
                "diarrhoea",
                "incontinence",
                "dyspnea",
                "chest_tightness_v5",
                "cough_v5",
                "wheezing_expiratory_distre",
                "stridor_inspiratory",
                "respiratory_arrest",
                "hypotension_collapse_v5",
                "dizziness",
                "tachycardia",
                "palpitations_cardiac_arryt",
                "chest_pain_angina_v5",
                "reductions_of_alertness",
                "loss_of_consciousness",
                "cardiac_arrest",
                "kind",
                "d_elicitor_gr5",
                "actual_vas_score",
                "vas_score_ann"};

        List<String[]> csvOutputData = new ArrayList<>();
        csvOutputData.add(header);
        for (Map<String, Object> a : cases.values()) {
            List<String> sample = sampleAsList(a);
            sample.add("" + a.get("vas_score_ann"));
            csvOutputData.add(sample.toArray(new String[0]));
        }
        writeCsvOutput(csvOutputData, dataDirectoryPath, fileName);
    }

    private static List<String> sampleAsList(Map<String, Object> verificationData) {
        List<String> sample = new ArrayList<>();
        sample.add("" + verificationData.get("case_id"));
        sample.add("" + verificationData.get("skin"));
        sample.add("" + verificationData.get("q_111_angioedema"));
        sample.add("" + verificationData.get("pharynx_larynx"));
        sample.add("" + verificationData.get("abdomin"));
        sample.add("" + verificationData.get("nausea"));
        sample.add("" + verificationData.get("vomiting"));
        sample.add("" + verificationData.get("diarrhoea"));
        sample.add("" + verificationData.get("incontinence"));
        sample.add("" + verificationData.get("dyspnea"));
        sample.add("" + verificationData.get("chest_tightness_v5"));
        sample.add("" + verificationData.get("cough_v5"));
        sample.add("" + verificationData.get("wheezing_expiratory_distre"));
        sample.add("" + verificationData.get("stridor_inspiratory"));
        sample.add("" + verificationData.get("respiratory_arrest"));
        sample.add("" + verificationData.get("hypotension_collapse_v5"));
        sample.add("" + verificationData.get("dizziness"));
        sample.add("" + verificationData.get("tachycardia"));
        sample.add("" + verificationData.get("palpitations_cardiac_arryt"));
        sample.add("" + verificationData.get("chest_pain_angina_v5"));
        sample.add("" + verificationData.get("reductions_of_alertness"));
        sample.add("" + verificationData.get("loss_of_consciousness"));
        sample.add("" + verificationData.get("cardiac_arrest"));
        sample.add("" + verificationData.get("kind"));
        sample.add("" + verificationData.get("d_elicitor_gr5"));
        sample.add("" + verificationData.get("actual_vas_score"));
        return sample;
    }

    private static void writeCsvOutput(List<String[]> csvOutputData, String dataDirectoryPath, String fileName) {
        File outputFile = new File(dataDirectoryPath + "/output/" + fileName);
        try (
            FileWriter fileWriter = new FileWriter(outputFile)) {
            CSVWriter writer = new CSVWriter(fileWriter, ',', CSVWriter.NO_QUOTE_CHARACTER);
            writer.writeAll(csvOutputData);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static void setFittedClassifiers(INDArray output, Map<Integer, Map<String, Object>> vasScores) {
        for (int i = 0; i < output.rows(); i++) {

            // set the classification from the fitted results
            vasScores.get(i).put("vas_score_ann",
                    vas.get(maxIndex(getFloatArrayFromSlice(output.slice(i)))));
        }
    }

    private static void setCaseIdAndActualClassifiert(Map<String, Pair<String, Integer>> answersToCaseId, Map<Integer, Map<String, Object>> vasScores) {
        for (int i = 0; i < vasScores.keySet().size(); i++) {

            Map<String, Object> verificationSample = vasScores.get(i);
            String answerKey = toAnswerHash(verificationSample);

            String caseId = answersToCaseId.get(answerKey).getLeft();
            Integer actualVasScore = answersToCaseId.get(answerKey).getRight();

            // set the classification from the training data
            verificationSample.put("actual_vas_score", actualVasScore);
            verificationSample.put("case_id", caseId);
        }
    }

    private static void setCaseIdOfFinalRegisterData(Map<Integer, String> answersToCaseId, Map<Integer, Map<String, Object>> finalRegisterData) {
        for (int i = 0; i < finalRegisterData.keySet().size(); i++) {
            Map<String, Object> finalRegisterRow = finalRegisterData.get(i);
            String caseId = answersToCaseId.get(i);
            finalRegisterRow.put("case_id", caseId);
        }
    }

    private static String toAnswerHash(Map<String, Object> verificationSample) {
        List<String> sampleAsList = sampleAsList(verificationSample);
        return Main.toAnswerHashKey(sampleAsList);
    }

    /**
     * This method is to show how to convert the INDArray to a float array. This is to
     * provide some more examples on how to convert INDArray to types that are more java
     * centric.
     */
    private static float[] getFloatArrayFromSlice(INDArray rowSlice) {
        float[] result = new float[rowSlice.columns()];
        for (int i = 0; i < rowSlice.columns(); i++) {
            result[i] = rowSlice.getFloat(i);
        }
        return result;
    }

    /**
     * find the maximum item index. This is used when the data is fitted and we
     * want to determine which class to assign the test row to
     */
    private static int maxIndex(float[] vals) {
        int maxIndex = 0;
        for (int i = 1; i < vals.length; i++) {
            float newnumber = vals[i];
            if ((newnumber > vals[maxIndex])) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * take the dataset loaded for the metric and make the record model out of it so
     * we can correlate the fitted classifier to the record.
     */
    private static Map<Integer, Map<String, Object>> makeRegisterForTesting(DataSet testData) {
        Map<Integer, Map<String, Object>> cases = new HashMap<>();

        INDArray features = testData.getFeatures();
        for (int i = 0; i < features.rows(); i++) {
            INDArray slice = features.slice(i);
            Map<String, Object> dataSample = new HashMap<>();

            //set the attributes
            setAttribute20191229(slice, dataSample);

            if (dataSample.entrySet().isEmpty()) {
                throw new RuntimeException("Missing configuration for column count: " + slice.columns());
            }

            cases.put(i, dataSample);
        }
        return cases;
    }

    private static void setAttribute20191229(INDArray slice, Map<String, Object> dataSample) {
        dataSample.put("skin", yesNo.get(slice.getInt(0)));
        dataSample.put("q_111_angioedema", yesNoUnknown.get((slice.getInt(1))));
        dataSample.put("pharynx_larynx", yesNo.get(slice.getInt(2)));
        dataSample.put("abdomin", yesNo.get(slice.getInt(3)));
        dataSample.put("nausea", yesNoUnknown.get((slice.getInt(4))));
        dataSample.put("vomiting", yesNoUnknown.get((slice.getInt(5))));
        dataSample.put("diarrhoea", yesNoUnknown.get((slice.getInt(6))));
        dataSample.put("incontinence", yesNoUnknown.get((slice.getInt(7))));
        dataSample.put("dyspnea", yesNoUnknown.get((slice.getInt(8))));
        dataSample.put("chest_tightness_v5", yesNoUnknown.get((slice.getInt(9))));
        dataSample.put("cough_v5", yesNoUnknown.get((slice.getInt(10))));
        dataSample.put("wheezing_expiratory_distre", yesNoUnknown.get((slice.getInt(11))));
        dataSample.put("stridor_inspiratory", yesNoUnknown.get((slice.getInt(12))));
        dataSample.put("respiratory_arrest", yesNoUnknown.get((slice.getInt(13))));
        dataSample.put("hypotension_collapse_v5", yesNoUnknown.get((slice.getInt(14))));
        dataSample.put("dizziness", yesNoUnknown.get((slice.getInt(15))));
        dataSample.put("tachycardia", yesNoUnknown.get((slice.getInt(16))));
        dataSample.put("palpitations_cardiac_arryt", yesNoUnknown.get((slice.getInt(17))));
        dataSample.put("chest_pain_angina_v5", yesNoUnknown.get((slice.getInt(18))));
        dataSample.put("reductions_of_alertness", yesNoUnknown.get((slice.getInt(19))));
        dataSample.put("loss_of_consciousness", yesNoUnknown.get((slice.getInt(20))));
        dataSample.put("cardiac_arrest", yesNoUnknown.get((slice.getInt(21))));
        dataSample.put("kind", yesNo.get(slice.getInt(22)));
        dataSample.put("d_elicitor_gr5", elicitors.get(slice.getInt(23)));
    }


    static Map<Integer, String> readEnumCSV(String csvFileClasspath) {
        try {
            List<String> lines = IOUtils.readLines(new ClassPathResource(csvFileClasspath).getInputStream(), StandardCharsets.UTF_8);
            Map<Integer, String> enums = new HashMap<>();
            for (String line : lines) {
                String[] parts = line.split(",");
                enums.put(Integer.parseInt(parts[0]), parts[1]);
            }
            return enums;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * used for testing and training
     */
    private static DataSet readCSVDataset(
            String fileSystemPath, int batchSize, int labelIndex, int numClasses)
            throws IOException, InterruptedException {

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(fileSystemPath)));
        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
        return iterator.next();
    }

}
