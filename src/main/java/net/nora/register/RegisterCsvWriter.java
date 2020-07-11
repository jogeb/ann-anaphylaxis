package net.nora.register;

import com.opencsv.CSVWriter;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UncheckedIOException;

public class RegisterCsvWriter {


    public void write(RegisterCsvData registerData, int trainingDataSize, String trainingDataFileName, String verificationDataFileName) {
        final int actualDataSize;
        if(trainingDataSize < 0){
            actualDataSize = registerData.getCsvData().size();
        }else{
            actualDataSize = trainingDataSize;
        }
        File train = new File(trainingDataFileName);
        write(registerData, train, 0, actualDataSize);
        File verification = new File(verificationDataFileName);
        if(trainingDataSize < 0){
            write(registerData, verification, 0, registerData.getCsvData().size());
        }else{
            write(registerData, verification, actualDataSize + 1, registerData.getCsvData().size());
        }
    }

    public void write(RegisterCsvData registerData, String fileName) {
        File dataToClassify = new File(fileName);
        try (FileWriter fileWriter = new FileWriter(dataToClassify)) {
            CSVWriter writer = new CSVWriter(fileWriter, ',', CSVWriter.NO_QUOTE_CHARACTER);
            writer.writeAll(registerData.getCsvData());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private void write(RegisterCsvData transformed, File train, int from, int to) {
        try (FileWriter fileWriter = new FileWriter(train)) {
            CSVWriter writer = new CSVWriter(fileWriter, ',', CSVWriter.NO_QUOTE_CHARACTER);
            writer.writeAll(transformed.getCsvData().subList(from, to));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

}
