import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.integer.ReplaceEmptyIntegerWithValueTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class Titanic {

    public static void main(String[] args) throws IOException, InterruptedException {
        RecordReader trainReader = new CSVRecordReader(1);
        RecordReader testReader = new CSVRecordReader(1);

        trainReader.initialize(new FileSplit(new File("train.csv")));
        testReader.initialize(new FileSplit(new File("test.csv")));

        Schema schema = new Schema.Builder()
                .addColumnInteger("PassengerId")
                .addColumnInteger("Survived")
                .addColumnInteger("Pclass")
                .addColumnString("Name")
                .addColumnCategorical("Sex", List.of("male","female"))
                .addColumnDouble("Age")
                .addColumnInteger("SibSp")
                .addColumnInteger("Parch")
                .addColumnString("Ticket")
                .addColumnDouble("Fare")
                .addColumnString("Cabin")
                .addColumnCategorical("Embarked", List.of("C", "Q", "S", ""))
                .build();

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("PassengerId", "Name", "Ticket", "Cabin")
                .categoricalToInteger("Sex")
                .categoricalToInteger("Embarked")
                .transform(new ReplaceEmptyIntegerWithValueTransform("Age",0))
                .transform(new ReplaceEmptyIntegerWithValueTransform("Fare", 0))
                .build();


        RecordReader trainTransform = new TransformProcessRecordReader(trainReader, transformProcess);
        RecordReader testTransform = new TransformProcessRecordReader(testReader, transformProcess);

        DataSetIterator trainIterator = new RecordReaderDataSetIterator(trainTransform, 32, 0,0, true);
        DataSetIterator testIterator = new RecordReaderDataSetIterator(testTransform, 32, 0,0, true);

        DataNormalization normalizationTrain = new NormalizerMinMaxScaler(-1,1);
        normalizationTrain.fit(trainIterator);

        DataNormalization normalizationTest = new NormalizerMinMaxScaler(-1,1);
        normalizationTest.fit(testIterator);

        trainIterator.setPreProcessor(normalizationTrain);
        testIterator.setPreProcessor(normalizationTest);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.RELU)
                .updater(new Adam(0.0003))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(7).nOut(64).build())
                .layer(1, new DenseLayer.Builder().nIn(64).nOut(64).build())
                .layer(2, new DenseLayer.Builder().nIn(64).nOut(32).build())
                .layer(3, new DenseLayer.Builder().nIn(32).nOut(16).build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(16).nOut(1).build())
                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        model.fit(trainIterator, 100);

        Evaluation evaluation = model.evaluate(testIterator);
        System.out.println(evaluation);

    }
}
