# ann-anaphylaxis
Java ANN built with deeplearning4j to classify severity of anaphylaxis case data

## Overview

This project contains the source code of the ann model that was used and described in the paper.
There is also some excerpt of the original training and register data that can be used to experiment with the code.
As well as the final trained ann model.

## Prerequisites

### 1. Make sure Java 8 or higher is installed

Run the following command to confirm:
```
$ java -version
java version "1.8.0_191"
Java(TM) SE Runtime Environment (build 1.8.0_191-b12)
Java HotSpot(TM) 64-Bit Server VM (build 25.191-b12, mixed mode)
```

### 2. Make sure Maven 3.6.0 or higher is installed
       
Run the following command to confirm:
```
$ mvn -version
Apache Maven 3.6.0 (97c98ec64a1fdfee7767ce5ffb20918da4f719f3; 2018-10-24T20:41:47+02:00)
Maven home: C:\Program Files (x86)\apache-maven-3.6.0\bin\..
Java version: 1.8.0_191, vendor: Oracle Corporation, runtime: C:\Program Files\Java\jdk1.8.0_191\jre
Default locale: de_DE, platform encoding: Cp1252
OS name: "windows 10", version: "10.0", arch: "amd64", family: "windows"
```

## Running the example

### 1. Compiling

Run the following command 
```
$ mvn clean compile
```

### 2. Training the model

Run the following command 
```
$ mvn test
```

## Experimenting with your own data

The project contains a small training dataset `data/input/testdata.csv` and an excerpt of cases of the register `data/input/register.csv`.


Training parameters can be configured in the `data/config.properties` file.

## Calculating vas values with an existing ann

The project also contains the ann which was trained and described in the paper `data/ann/trained.nnet`.
 When setting the property ann.use.saved.ann=true in `data/config.properties` the ann model is used to
  calculate vas values for the register file `data/input/register.csv`. You can also add own rows that 
  should be calculated by the ann. The result will be written to `data/output/ann_calculated.csv`.

## Adding own cases 
To add your own training data to `input/testdata.csv` or register data that should be classified by the ann to `input/register.csv`,
you can add new lines. The columns have the following semantics:

| Column                           | Description                                        | Value Type                       |
| -------------------------------- | -------------------------------------------------- | -------------------------------- |
| b_case_id                        | a case id to identify the line                     | number (must be unique)          |
| skin                             | skin symptoms (urticaria, itch, erythema or flush) | yes,no                           |
| q_111_angioedema                 | angioedema                                         | yes,no                           |
| pharynx_larynx                   | pharyngeal symptoms                                | yes,no                           |
| abdomin                          | abdominal pain / cramps/ distention                | yes,no                           |
| q_112_nausea                     | nausea                                             | yes,no                           |
| q_112_vomiting                   | vomiting                                           | yes,no                           |
| q_112_diarrhoea                  | diarrhoea                                          | yes,no                           |
| q_112_incontinence               | incontinence                                       | yes,no                           |
| q_113_dyspnea                    | dyspnea / shortness of breath                      | yes,no                           |
| q_113_chest_tightness_v5         | chest tightness                                    | yes,no                           |
| q_113_cough_v5                   | cough                                              | yes,no                           |
| q_113_wheezing_expiratory_distre | wheezing (expiratory)                              | yes,no                           |
| q_113_stridor_inspiratory        | stridor (inspiratory)                              | yes,no                           |
| q_113_respiratory_arrest         | respiratory arrest                                 | yes,no                           |
| q_114_hypotension_collapse_v5    | hypotension (collapse)                             | yes,no                           |
| q_114_dizziness                  | dizziness                                          | yes,no                           |
| q_114_tachycardia                | tachycardia                                        | yes,no                           |
| q_114_palpitations_cardiac_arryt | palpitations / arrhythmia                          | yes,no                           |
| q_114_chest_pain_angina_v5       | chest pain / angina                                | yes,no                           |
| q_114_reductions_of_alertness    | reduction of alertness                             | yes,no                           |
| q_114_loss_of_consciousness      | loss of consciousness                              | yes,no                           |
| q_114_cardiac_arrest             | cardiac arrest                                     | yes,no                           |
| kind                             | patient is a child (age < 13 y)                    | yes,no                           |
| d_elicitor_gr5                   | Elicitor: venom / food / drug / others / unknown   | other,unkown,drugs,insects,food  |
| VAS_MK                           | manual rated VAS score                             | number (1-10)                    |
| vas_score_ann                    | ANN rated VAS score                                | number (1-10), will be generated |

