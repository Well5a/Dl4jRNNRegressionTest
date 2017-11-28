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
import org.deeplearning4j.eval.RegressionEvaluation;
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
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Dl4jTestSingle 
{
	//Good parameters:
	//For sinus: 7000, 0.001, 50
	//For linreg: 8000, 0.001, 50
	//For passengers: 3000, 0.001, 10
	static String model = "sinus"; //sinus | linreg | passengers
	static int nEpochs = 7000;
	static double learningrate = 0.001;
	static int nHidden = 50;
	
    @Test
    public void predict() throws Exception 
    {
    	int batchSize = 32;    	
    	File baseDir = new ClassPathResource("").getFile();
    	
		SequenceRecordReader trainReader = new CSVSequenceRecordReader(0, ","); //For passengers set delimiter to ";"
		trainReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/SingleTimestep/"+model+"_train_%d.csv", 0, 0));     //(new FileSplit(new ClassPathResource("training_0.csv").getFile()));
	    //For regression, numPossibleLabels is not used. Setting it to -1 here
	    DataSetIterator trainIter = new SequenceRecordReaderDataSetIterator(trainReader, batchSize, -1, 1, true); //reader, batch size, label index,number of possible labels, regression	    
	    DataSet trainData = trainIter.next();
	    
	    SequenceRecordReader testReader = new CSVSequenceRecordReader(0, ","); //For passengers set delimiter to ";"
	    testReader.initialize(new NumberedFileInputSplit(baseDir.getAbsolutePath() + "/SingleTimestep/"+model+"_test_%d.csv", 0, 0));	    
	    DataSetIterator testIter = new SequenceRecordReaderDataSetIterator(testReader, batchSize, -1, 1, true); //reader, batch size, label index,number of possible labels, regression
	    DataSet testData = testIter.next();
        
	    //Normalize the training data
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);
        normalizer.fit(trainData);  
        
        normalizer.transform(trainData);
        normalizer.transform(testData);
        
	    System.out.println(trainData.toString());
	    System.out.println(testData.toString());

        
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
//	    		             .backpropType(BackpropType.TruncatedBPTT)
//	    		             .tBPTTForwardLength(10)
//	    		             .tBPTTBackwardLength(10)
	    		             .build();
	    
	    
	    MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        
        net.setListeners(new ScoreIterationListener(20));

        // ----- Train the network, evaluating the test set performance at each epoch -----
       

        System.out.println("----- Training Start -----");
        for (int i = 0; i < nEpochs; i++) 
        {
            net.fit(trainData);
            
            if(i % 100 == 0)
            {
            	System.out.println("----- Still training: "+i+" -----");
            	//Run regression evaluation on our single column input
                RegressionEvaluation evaluation = new RegressionEvaluation(1);
                INDArray features = testData.getFeatureMatrix();

                INDArray lables = testData.getLabels();
                INDArray predicted = net.output(features, false);

                evaluation.evalTimeSeries(lables, predicted);
                
                //Just do sout here since the logger will shift the shift the columns of the stats
                System.out.println(evaluation.stats());
            }
        }
        
        System.out.println("----- Training Complete -----");
             
        
        //Init rrnTimeStep with train data and predict test data
        INDArray init = net.rnnTimeStep(trainData.getFeatureMatrix());
        
        //DataSet t = testIter.next();
        INDArray predicted = net.rnnTimeStep(testData.getFeatureMatrix());

        //Revert data back to original values for plotting
        normalizer.revert(trainData);
        normalizer.revertLabels(init);
        normalizer.revert(testData);
        normalizer.revertLabels(predicted);

        INDArray trainFeatures = trainData.getLabels();
        INDArray testFeatures = testData.getLabels();
        
        System.out.println(trainData);
        System.out.println(testData);
        System.out.println(predicted);
        
        //Create plot with out data
        XYSeriesCollection c = new XYSeriesCollection();
        createSeries(c, trainFeatures, 0, "Train data");
        createSeries(c, init, 0, "Initial Train data");
        createSeries(c, testFeatures, 100, "Actual test data");
        createSeries(c, predicted, 100, "Predicted test data");

        plotDataset(c);

        System.out.println("----- Example Complete -----");
        for(;;);
    }
    
    private static void createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name) {
        int nRows = data.shape()[2];
        XYSeries series = new XYSeries(name);
        for (int i = 0; i < nRows; i++) {
            series.add(i + offset, data.getDouble(i));
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
       
        BufferedImage image = new BufferedImage(f.getWidth(), f.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics2D = image.createGraphics();
        f.paint(graphics2D);
        String ext = String.valueOf(nEpochs + learningrate);
        ImageIO.write(image,"jpeg", new File("/home/ubuntu/Schreibtisch/Dl4j Test Graphs/SingleTimestep/"+model+"/hl"+nHidden+"/regression-"+ext));
    }
}
