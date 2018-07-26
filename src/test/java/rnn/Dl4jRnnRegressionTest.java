package rnn;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import javax.swing.*;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.List;


/**
 * This unit test shows how to predict time series data with Dl4j recurrent neural networks.  
 * Specify the static parameters and run the test. The plot helps you to evaluate how well the prediction went.
 * Be aware that you have to stop the JUnit Test manually because of an endless for loop to hold the plot.
 * The plot is then stored at the specified directory <tt>imageSaveDir</tt>.
 * <p>
 * The data is loaded from one CSV file (e.g. linreg_raw.csv). This file is split in a train and a test set. 
 * These sets again are split into several smaller time series. One series consists of features (their number is specified by <tt>minibatchSize</tt>) and one label.
 * Every feature and label represents a single time step in the series, in which the label is the last one and the value to be predicted. 
 * 
 * Good parameters: nEpochs	learnRate  	nHidden
 * For linreg: 		75	 	0.2			10 		(gets better with more epochs)
 * For sinus: 		50		0.15		50 		(gets better with more hidden layers)
 * For passengers: 	100		0.3			10 		(gets better with more epochs)
 */

public class Dl4jRnnRegressionTest 
{
    private static File initBaseFile(String fileName) {
        try {
            return new ClassPathResource(fileName).getFile();
        } catch (IOException e) {
            throw new Error(e);
        }
    }

    private static File baseDir 			= initBaseFile("");
    private static File baseDirTrain 		= new File(baseDir, "multiTimestepTrain");
    private static File featuresDirTrain 	= new File(baseDirTrain, "features");
    private static File labelsDirTrain 		= new File(baseDirTrain, "labels");
    private static File baseDirTest 		= new File(baseDir, "multiTimestepTest");
    private static File featuresDirTest 	= new File(baseDirTest, "features");
    private static File labelsDirTest 		= new File(baseDirTest, "labels");

    private static int numVariables = 0;  // in csv.
    
    //The dataset to be used: linreg | passengers | sinus
    private final static String datasetName = "linreg";
    
    //include the series of time values in training or not
    private final static boolean includeTimeData = false;
    
    //Set number of examples for training, testing, and time steps
    private final static int trainSize 		= 100;		//max 100 for passengers, sinus - max 800 for linreg
    private final static int testSize 		= 40; 		//max 20 for passengers - max 40 for sinus - max 400 for linreg
    private final static int miniBatchSize 	= 20;
        
    //Hyperparameters for tuning
    private final static int nEpochs 		= 50;		//number of iterations
	private final static double learnRate   = 1e-6;		//1e-6
    private final static double l2Value     = 1e-6;
    private final static double dropoutRate = 0.8;		//probability of retaining an activation
    private final static int backPropLength = 20;
    private final static int nHidden 		= 25;		//number of hidden layers
    
	//Location of the saved plots
    private final static String imageSaveDir = "C:\\Users\\mwe\\Desktop\\Dl4j Test Graphs\\"+datasetName+"\\hl"+nHidden+"\\regression-"+String.valueOf(nEpochs + learnRate)+".jpeg";

	@Test
    public void predict() throws Exception 
    {
        /**
         * Monitoring and Visualization
         */
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);


        /**
         * Data Preprocessing
         */
        //Prepare multi time step data, see method comments for more info
        List<String> rawStrings = prepareTrainAndTest(trainSize, testSize, miniBatchSize);
        System.out.println(rawStrings.toString());

        
        // ----- Load the training data -----
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "\\train_%d.csv", 0, trainSize - 1));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "\\train_%d.csv", 0, trainSize - 1));

        DataSetIterator trainDataIter = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, 1, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        // ----- Load the test data -----
        //Same process as for the training data.
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "\\test_%d.csv", trainSize, trainSize + testSize - 1));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "\\test_%d.csv", trainSize, trainSize + testSize - 1));
  
        DataSetIterator testDataIter = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, 1, -1, true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
            
        
        //Normalize the training data
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);
        normalizer.fit(trainDataIter);              //Collect training data statistics
        trainDataIter.reset();
        trainDataIter.setPreProcessor(normalizer);
        testDataIter.setPreProcessor(normalizer);
        
        
        // ----- Configure the network -----
        MultiLayerConfiguration netConf = new NeuralNetConfiguration.Builder()
            .seed(140)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(learnRate))
            .l2(l2Value)
            //.dropOut(dropoutRate)
            .list()
            .layer(0, new LSTM.Builder()
            	.activation(Activation.SOFTSIGN).nIn(numVariables).nOut(nHidden)
                .build())
            .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY).nIn(nHidden).nOut(numVariables)
                .build())
            .backpropType(BackpropType.TruncatedBPTT)
            .tBPTTLength(backPropLength)
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(netConf);
        net.init();
        net.setListeners(new ScoreIterationListener(20));

        //Then add the StatsListener to collect this information from the network, as it trains
        net.setListeners(new StatsListener(statsStorage));


        // ----- Train the network -----
        for (int i = 0; i < nEpochs; i++)
        {
            net.fit(trainDataIter);
            trainDataIter.reset();

            System.out.println("Epoch: "+(i+1)+" / "+nEpochs+"\n");
        }

        System.out.println(net.score());
        
//        @SuppressWarnings({ "rawtypes", "unchecked" })
//		EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
////        		.epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(3))
////        		.iterationTerminationConditions(new MaxTimeIterationTerminationCondition(3, TimeUnit.MINUTES))
//        		.epochTerminationConditions(new MaxEpochsTerminationCondition(nEpochs))
//        		.scoreCalculator(new DataSetLossCalculator(testDataIter, true))
//                .evaluateEveryNEpochs(1)
//        		.modelSaver(new LocalFileModelSaver(""))
//        		.build();
//
//        @SuppressWarnings("unchecked")
//		EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, netConf , trainDataIter);
//
//
//
//        System.out.println("---- Start Training with "+nEpochs+" Epochs ----");
//        //Conduct early stopping training:
//        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
//
//        System.out.println(result);
//
//        //Print out the results:
//        System.out.println("Termination reason: " + result.getTerminationReason());
//        System.out.println("Termination details: " + result.getTerminationDetails());
//        System.out.println("Total epochs: " + result.getTotalEpochs());
//        System.out.println("Best epoch number: " + result.getBestModelEpoch());
//        System.out.println("Score at best epoch: " + result.getBestModelScore());
//
//        //Get the best model:
//        MultiLayerNetwork net = result.getBestModel();


        /*
         * All code below this point is only necessary for plotting
         */
        trainDataIter.reset();
        testDataIter.reset();

        //Convert raw string data to IndArrays for plotting
        INDArray trainArray = createIndArrayFromStringList(rawStrings, 0, trainSize);
        INDArray testArray 	= createIndArrayFromStringList(rawStrings, trainSize, testSize);

        //Create plot for the actual data
        XYSeriesCollection c = new XYSeriesCollection();
        createSeries(c, trainArray, 0, "Train data");
        createSeries(c, testArray, trainSize, "Actual test data");
        
        
        //Init rnnTimeStep with train data, creating a plot for every step
        int timeStepCount = 0;
        INDArray init = Nd4j.zeros(miniBatchSize, 1);
        trainDataIter.next();
        trainDataIter.next();
        while (trainDataIter.hasNext())
        {
        	init = net.rnnTimeStep(trainDataIter.next().getFeatureMatrix());
        	normalizer.revertLabels(init);
            createSeries(c, init, timeStepCount*miniBatchSize + 2*miniBatchSize, "train " + String.valueOf(timeStepCount));
            timeStepCount++;
        }
        trainDataIter.reset();

        
        //Predict on the test data, creating a plot for every step
        timeStepCount = 0;
        INDArray predicted = Nd4j.zeros(miniBatchSize, 1);
        while (testDataIter.hasNext()) 
        {
        	predicted = net.rnnTimeStep(testDataIter.next().getFeatureMatrix());
        	normalizer.revertLabels(predicted);
        	System.out.println(predicted);
            createSeries(c, predicted, timeStepCount*miniBatchSize+trainSize, "predict " + String.valueOf(timeStepCount));
            timeStepCount++;
        }
        testDataIter.reset();

        plotDataset(c);

        System.out.print("----- Example Complete -----");
        
        for(;;); //hold the plot
    }


    /**
     * Creates an IndArray from a list of strings
     * Used for plotting purposes
     */
    private static INDArray createIndArrayFromStringList(List<String> rawStrings, int startIndex, int length) 
    {
        List<String> stringList = rawStrings.subList(startIndex, startIndex + length);

        double[][] primitives = new double[numVariables][stringList.size()];
        
        for (int i = 0; i < stringList.size(); i++) 
        {
            String[] vals = stringList.get(i).split(",");
            
            for (int j = 0; j < vals.length; j++)
            {
                primitives[j][i] = Double.valueOf(vals[j]);
            }
        }

        return Nd4j.create(new int[]{1, length}, primitives);
    }
    

    /**
     * Used to create the different time series for plotting purposes
     */
    private static void createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name)
    {
        int nRows = data.shape()[2];
        //System.out.println("nRows: " + nRows);

        System.out.println(name+": "+offset);
        System.out.println(data.slice(0));

        XYSeries series = new XYSeries(name);
        
        //System.out.println(data);
        
//        if(includeTimeData)
//        {
//        	  for (int i = 0; i < nRows; i++)
//              {
//                  if (name.startsWith("predict") || name.startsWith("train") )
//                  {
//                      series.add(data.slice(0).getDouble(nRows + i), data.slice(0).getDouble(i));
//                  }
//                  else
//                  {
//                      series.add(data.slice(1).getDouble(i), data.slice(0).getDouble(i));
//                  }
//              }
//        }
//        else
//        {
//        	 for (int i = 0; i < nRows; i++)
//             {
//                 series.add(i + offset, data.slice(0).getDouble(i));
//             }
//
//        }

        series.add(offset, data.slice(0).getDouble(0));

        seriesCollection.addSeries(series);
    }
    

    /**
     * Generate an xy plot of the datasets provided.
     */
    private static void plotDataset(XYSeriesCollection c) throws IOException 
    {
    	System.out.println(c.getSeries(2).getItems().toString());
        String title = "RNN Regression Test for "+datasetName+" dataset";
        String xAxisLabel = "Timestep";
        String yAxisLabel = "Count";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);
        
        // get a reference to the plot for further customization...
        final XYPlot plot = chart.getXYPlot();

        // Auto zoom to fit time series in initial window
        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);

        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        RefineryUtilities.centerFrameOnScreen(f);
        f.setVisible(true);
        
        //save the plot
        BufferedImage image = new BufferedImage(f.getWidth(), f.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics2D = image.createGraphics();
        f.paint(graphics2D);
        ImageIO.write(image,"jpeg", new File(imageSaveDir));
    }

    
    /**
     * Converts data from a CSV file to a structure that can be used for a multi time step problem. 
     */
    private static List<String> prepareTrainAndTest(int trainSize, int testSize, int numberOfTimesteps) throws IOException 
    {
        Path rawPath = Paths.get(baseDir.getAbsolutePath() + "\\"+datasetName+"_raw.csv");

        List<String> rawStrings = Files.readAllLines(rawPath, Charset.defaultCharset());
        
        if(!includeTimeData)
        {
            for(int i = 0; i < rawStrings.size(); i++)
            {
            	rawStrings.set(i, rawStrings.get(i).split(",")[1]);
            }
        }
        else
        {
			for(int i = 0; i < rawStrings.size(); i++)
		    {
				rawStrings.set(i, rawStrings.get(i).split(",")[1] + "," + rawStrings.get(i).split(",")[0]);
		    }
        }
        
        setNumOfVariables(rawStrings);

        //Remove all files before generating new ones
        FileUtils.cleanDirectory(featuresDirTrain);
        FileUtils.cleanDirectory(labelsDirTrain);
        FileUtils.cleanDirectory(featuresDirTest);
        FileUtils.cleanDirectory(labelsDirTest);

        int fileWriteCounter = 0;
        
        for (int i = 0; i < trainSize; i++) 
        {
            Path featuresPath = Paths.get(featuresDirTrain.getAbsolutePath() + "\\train_" + i + ".csv");
            Path labelsPath = Paths.get(labelsDirTrain + "\\train_" + i + ".csv");
            
            for (int j = 0; j < numberOfTimesteps; j++) 
            {
                Files.write(featuresPath, rawStrings.get(i + j).concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
            }
            
            Files.write(labelsPath, rawStrings.get(i + numberOfTimesteps).concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
            System.out.println("Writing Train Files " + (++fileWriteCounter) + " \\ " + (trainSize));
        }

        fileWriteCounter = 0;
        
        for (int i = trainSize; i < testSize + trainSize; i++)
        {
            Path featuresPath = Paths.get(featuresDirTest + "\\test_" + i + ".csv");
            Path labelsPath = Paths.get(labelsDirTest + "\\test_" + i + ".csv");
            
            for (int j = 0; j < numberOfTimesteps; j++) 
            {
                Files.write(featuresPath, rawStrings.get(i + j).concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
            }

            Files.write(labelsPath, rawStrings.get(i + numberOfTimesteps).concat(System.lineSeparator()).getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
            System.out.println("Writing Test Files " + (++fileWriteCounter) + " \\ " + (testSize));
        }
        
        return rawStrings;
    }
    

    private static void setNumOfVariables(List<String> rawStrings) 
    {
        numVariables = rawStrings.get(0).split(",").length;
    }
}