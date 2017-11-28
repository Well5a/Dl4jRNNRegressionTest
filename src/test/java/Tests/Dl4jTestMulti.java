package Tests;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * This unit test shows how to predict time series data with Dl4j recurrent neural networks.  
 * Specify the static parameters and run the test. The plot helps you to evaluate how well the prediction worked.
 * Be aware that you have to stop the JUnit Test manually because of an endless for loop to hold the plot.
 * The plot is then stored at the specified directory <tt>imageSaveDir</tt>.
 * <p>
 * The data is loaded from several CSV files. Every file is an own time series which is trained or predicted individually.
 * 
 * 
 * @author mwe
 */
public class Dl4jTestMulti 
{
	static String datasetName 	= "linreg"; //sinus | linreg | passengers
	static int nEpochs 			= 1000;		//number of iterations
	static double learningrate 	= 0.001;	//higher values like '0.01' can't predict values (!?)
	static int nHidden 			= 50;		//number of hidden layers
	static int miniBatchSize 	= 32;  		//seems not to have any effect
	
	static int nTrainValues		= 100; 		//number of values for training, used for plotting
	static String imageSaveDir 	= "/home/ubuntu/Schreibtisch/Dl4j Test Graphs/MultiTimestep/"+datasetName+"/hl"+nHidden+"/regression-"+String.valueOf(nEpochs + learningrate);
	
    @Test
    public void predict() throws Exception 
    {
    	// ----- Load train and test data -----
    	File baseDir 		= new ClassPathResource("").getFile();
    	String delimiter 	= ",";
    	if(datasetName.equals("passengers")) delimiter = ";";
    	
		SequenceRecordReader trainReader = new CSVSequenceRecordReader(0, delimiter);
		trainReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/MultiTimestep/"+datasetName+"_train_%d.csv", 0, 4));   //loads 5 files
//		trainReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/MultiTimestep/alternative/"+model+"_train_%d.csv", 0, 1));  //loads 2 files
//		trainReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/SingleTimestep/"+model+"_train_%d.csv", 0, 0));   //loads 1 file
		//For regression, numPossibleLabels is not used. Setting it to -1 here
	    DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(trainReader, miniBatchSize, -1, 1, true); //reader, batch size, label index,number of possible labels, regression	
	    DataSet trainData = trainIter.next();
	    
	    SequenceRecordReader testReader = new CSVSequenceRecordReader(0, delimiter);
	    testReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/MultiTimestep/"+datasetName+"_test_%d.csv", 0, 4));	//loads 5 files    
	    DataSetIterator testIter = new SequenceRecordReaderDataSetIterator(testReader, miniBatchSize, -1, 1, true); //reader, batch size, label index,number of possible labels, regression
        DataSet testData = testIter.next();
	    
        // ----- Output loaded data -----
        System.out.println("----- train data: ----- \n" + trainData);
        System.out.println("----- test data: ----- \n" + testData);
        
	    // ----- Normalize the train and test data -----
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1); //Normalize between 0 and 1
        normalizer.fitLabel(true);
        normalizer.fit(trainData);  
        
        normalizer.transform(trainData);
        normalizer.transform(testData); //test data is >1

        // ----- Output normalized data -----
        System.out.println("----- normalized train data: ----- \n" + trainData);
        System.out.println("----- normalized test data: ----- \n" + testData);
        
        // ----- Configure the neural network -----
	    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	    		             .seed(140)
	    		             .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	    		             .iterations(1)
	    		             .weightInit(WeightInit.XAVIER)
	    		             .updater(Updater.NESTEROVS)
	    		             .learningRate(learningrate)
	    		             .list()
	    		             .layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(nHidden)
	    		                 .build())
	    		             .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
	    		                 .activation(Activation.IDENTITY).nIn(nHidden).nOut(1).build())
	    		             .build();
	    	    
	    // ----- Initialize the network -----
	    MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(20));

        
        // ----- Train the network, evaluating the test set performance at each epoch -----
        System.out.println("----- Training Start -----");   
        for (int i = 0; i < nEpochs; i++) 
        {
        	trainData.forEach(timeSeries -> net.fit(timeSeries));
           
            if(i % 100 == 0)
            {
            	System.out.println("----- Still training: "+i+" -----");
            }
        }
        System.out.println("----- Training Complete -----");
     
        
        // ----- Initialize rrnTimeStep with train data and predict test data -----
        int seriesIter = 0;
        INDArray init = Nd4j.zeros(1, 1);
        for(DataSet timeSeries : trainData)
        {
        	if(seriesIter == 0) //for the first series
        	{
        		init = net.rnnTimeStep(timeSeries.getFeatureMatrix());
        	}
        	else
        	{
        		init = Nd4j.vstack(init, net.rnnTimeStep(timeSeries.getFeatureMatrix()));
        	}
        	seriesIter++;
        }
        System.out.println("----- initialized train data: ----- \n" + init);
        
        seriesIter = 0;
        INDArray predicted = Nd4j.zeros(1, 1);
        for(DataSet timeSeries : testData)
        {
        	if(seriesIter == 0)
        	{
        		predicted = net.rnnTimeStep(timeSeries.getFeatureMatrix());
        	}
        	else
        	{
        		predicted = Nd4j.vstack(predicted, net.rnnTimeStep(timeSeries.getFeatureMatrix()));
        	}
        	seriesIter++;
        }   
        System.out.println("----- predicted test data: ----- \n" + predicted);
        
        // ----- Revert data back to original values for plotting -----
        normalizer.revert(trainData);
        normalizer.revertLabels(init);
        normalizer.revert(testData);
        normalizer.revertLabels(predicted);

        INDArray trainFeatures = trainData.getLabels();
        INDArray testFeatures = testData.getLabels();
        
        // ----- Output the denormalized data -----
        System.out.println("----- denormalized initialized train data: ----- \n" + init);
        System.out.println("----- denormalized predicted test data: ----- \n" + predicted);
        
        
        // ----- Create plot -----
        XYSeriesCollection c = new XYSeriesCollection();
        createSeries(c, trainFeatures, 0, "Train data");
        createSeries(c, init, 0, "Initial Train data");
        createSeries(c, testFeatures, nTrainValues, "Actual test data");
        createSeries(c, predicted, nTrainValues, "Predicted test data");

        plotDataset(c);

        System.out.println("----- Example Complete -----");
        for(;;); //hold plot
    }
    
    private static void createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name) 
    {
        XYSeries series = new XYSeries(name);
        int nRows = data.size(0);
        int j = 0;
        for (int i = 0; i < nRows; i++) 
        {
        	INDArray row = data.getRow(i);
        	int nCols = row.size(1);
        	for(int k = 0; k < nCols; k++)
        	{
        		series.add(j + offset, row.getDouble(0, k));
                j++;
        	}
        }
        seriesCollection.addSeries(series);
    }
    
    private static void plotDataset(XYSeriesCollection c) throws IOException {

        String title = "Regression example";
        String xAxisLabel = "Timestep";
        String yAxisLabel = "Count";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);

        // get a reference to the plot for further customisation...
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
}
