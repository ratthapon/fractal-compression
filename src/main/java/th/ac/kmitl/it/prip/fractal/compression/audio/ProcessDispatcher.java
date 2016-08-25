package th.ac.kmitl.it.prip.fractal.compression.audio;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.sound.sampled.UnsupportedAudioFileException;

import th.ac.kmitl.it.prip.fractal.Parameters;
import th.ac.kmitl.it.prip.fractal.compression.audio.gpu.CUCompressor;
import th.ac.kmitl.it.prip.fractal.dataset.DataSetManager;

public class ProcessDispatcher {
	private static final Logger LOGGER = Logger.getLogger(ProcessDispatcher.class.getName());
	public static final String[] UNITS = { "", "k", "M", "G", "T", "P" };
	private static int DELTA_TIME = 5000;

	private AtomicInteger processedSamples = new AtomicInteger(0);
	private AtomicInteger processedParts = new AtomicInteger(0);
	private AtomicInteger passedNSamples = new AtomicInteger(0);
	private AtomicInteger passedNParts = new AtomicInteger(0);
	private AtomicInteger samplesCounter = new AtomicInteger(0);
	private AtomicInteger partsCounter = new AtomicInteger(0);
	private DataSetManager dataSetManager;
	private Parameters parameters;

	private static int samplesCompressSpeed; // samples per sec
	private static int partsCompressSpeed; // parts per sec

	private TimerTask estimateSpeed = new TimerTask() {

		@Override
		public void run() {

			samplesCompressSpeed = (samplesCounter.get() - passedNSamples.get()) / (DELTA_TIME / 1000);
			passedNSamples.set(samplesCounter.get());

			partsCompressSpeed = (partsCounter.get() - passedNParts.get()) / (DELTA_TIME / 1000);
			passedNParts.set(partsCounter.get());
		}
	};
	private TimerTask printState = new TimerTask() {

		@Override
		public void run() {
			LOGGER.log(Level.INFO,
					" Speed " + samplesCompressSpeed + " samples/sec " + partsCompressSpeed + " parts/sec");
		}
	};

	private void dispatch() throws IOException, InterruptedException {
		// single process executor
		ExecutorService executorService = Executors.newFixedThreadPool(1);
		List<Callable<double[][]>> compressorQueue = new ArrayList<Callable<double[][]>>();

		for (int i = 0; i < dataSetManager.getSize(); i++) {
			final int idsIdx = i;
			if (dataSetManager.existOutput(idsIdx) && parameters.isSkipIfExist()) {
				String log = "Skiped " + idsIdx + " " + dataSetManager.getName(idsIdx);
				LOGGER.log(Level.INFO, log);
			} else {
				compressorQueue.add(new Callable<double[][]>() {

					@Override
					public double[][] call() throws Exception {
						double[][] codes = null;
						float[] inputAudioData = dataSetManager.readAudio(idsIdx);
						if (parameters.isGpuEnable()) {
							// process compressor
							CUCompressor compressor = new CUCompressor(inputAudioData, parameters);
							compressor.registerPartCount(partsCounter);
							compressor.registerSampleCount(samplesCounter);
							codes = compressor.process();
							// logging

							writeLogs(compressor, idsIdx);
						} else {
							Compressor compressor = new Compressor(inputAudioData, parameters);
							compressor.registerPartCount(partsCounter);
							compressor.registerSampleCount(samplesCounter);
							codes = compressor.process();
							// logging
							writeLogs(compressor, idsIdx);
						}

						// store minimum value of self similarity
						dataSetManager.writeCode(idsIdx, codes);
						return codes;
					}
				});
			}
		}
		try {
			executorService.invokeAll(compressorQueue);
		} catch (InterruptedException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		} finally {
			executorService.shutdown();
		}
	}

	private void prepare() throws IOException {
		// generate output directory
		Paths.get(parameters.getOutdir()).toFile().mkdirs();

		// generate parameter setting
		List<String> info = new ArrayList<String>();
		List<String> paramsList;
		// save process description
		try {
			info.add("Java " + System.getProperty("java.version"));
			info.add("Version " + ProcessDispatcher.class.getPackage().getImplementationVersion());
			info.add("Date " + LocalDateTime.now());
			info.add(parameters.toString());

			paramsList = new ArrayList<String>(Arrays.asList(parameters.getInArgs()));

			Files.write(Paths.get(parameters.getOutdir(), "\\info.txt"), info);
			Files.write(Paths.get(parameters.getOutdir(), "\\parameters.txt"), paramsList);
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}

		// if help
		if (parameters.isHelp()) {
			try {
				InputStream in = ProcessDispatcher.class.getResourceAsStream("help.txt");
				BufferedReader br = new BufferedReader(new InputStreamReader(in));
				String help = br.readLine();
				while (help.length() > 0) {
					LOGGER.log(Level.INFO, help);
					help = br.readLine();
				}
				in.close();
				br.close();
			} catch (IOException e) {
				LOGGER.log(Level.SEVERE, e.getMessage());
				throw e;
			}
		}

		// validate parameters
		if (!parameters.isValidParams()) {
			LOGGER.log(Level.INFO, "Incorrect parameters");
		}
	}

	private void estimate() throws IOException, UnsupportedAudioFileException {
		int nSamples = dataSetManager.estimateNumSamples();
		int samplesUnit = (int) (Math.log10(nSamples) / 3);
		LOGGER.log(Level.INFO, String.format("Expect %d %s samples", (int) (nSamples / Math.pow(10, samplesUnit * 3)),
				UNITS[samplesUnit]));
	}

	public void exec() throws IOException, UnsupportedAudioFileException, InterruptedException {
		Timer speedEstimator = new Timer();
		Timer printTimer = null;
		try {
			LOGGER.log(Level.INFO, "Test name " + parameters.getTestName());
			LOGGER.log(Level.INFO, parameters.toString());
			if (parameters.isValidParams()) {
				prepare();
				estimate();

				// start speed estimation
				speedEstimator.scheduleAtFixedRate(estimateSpeed, 0, DELTA_TIME);

				// progress report
				if (parameters.getProgressReportRate() > 0) {
					printTimer = new Timer();
					DELTA_TIME = (int) parameters.getProgressReportRate();
					printTimer.scheduleAtFixedRate(printState, 0, DELTA_TIME);
				}

				dispatch();
			}
		} catch (IOException | UnsupportedAudioFileException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
		speedEstimator.cancel();
		if (printTimer != null) {
			printTimer.cancel();
		}
	}

	private void writeLogs(Compressor compressor, final int idsIdx) throws IOException {
		// logging
		String log = "Compressed " + idsIdx + " " + dataSetManager.getName(idsIdx) + " progress "
				+ compressor.getNSamples() + "/" + compressor.getNSamples() + " part " + compressor.getNParts() + "/"
				+ compressor.getNParts() + " time " + compressor.time() / 1000 + " sec";
		LOGGER.log(Level.INFO, log);
		dataSetManager.updateTimingLog(idsIdx, compressor.time());
		dataSetManager.updateCompletedLog(idsIdx, log);
	}

	public ProcessDispatcher(Parameters parameters) throws IOException {
		this.dataSetManager = new DataSetManager(parameters);
		this.parameters = parameters;
	}

	public AtomicInteger getProcessedSamples() {
		return processedSamples;
	}

	public AtomicInteger getProcessedParts() {
		return processedParts;
	}

	public AtomicInteger getPassedNSamples() {
		return passedNSamples;
	}

	public AtomicInteger getPassedNParts() {
		return passedNParts;
	}
}
