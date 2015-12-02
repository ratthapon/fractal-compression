package th.ac.kmitl.it.prip.fractal.compression.audio;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

import th.ac.kmitl.it.prip.fractal.DataHandler;
import th.ac.kmitl.it.prip.fractal.Executer;

public class CompressorExecuter extends Executer {
	private static AtomicInteger processedSamples = new AtomicInteger(0);
	private static AtomicInteger processedParts = new AtomicInteger(0);
	private static AtomicInteger passedNSamples = new AtomicInteger(0);
	private static AtomicInteger passedNParts = new AtomicInteger(0);
	private static int samplesCompressSpeed; // samples per sec
	private static int partsCompressSpeed; // parts per sec

	private static void estimate() {
		// estimate runtime
		final String[] idsList = DataHandler.getIdsPathList(parameters);
		for (int i = 0; i < idsList.length; i++) {
			float[] inputAudioData = DataHandler.audioread(idsList[i],
					parameters.getInExtension());
			Compressor compressor = new Compressor(inputAudioData, parameters);
			nSamples += compressor.getNSamples();
			nParts += compressor.getNParts();
		}
		int samplesUnit = (int) (Math.log10(nSamples) / 3);
		int partsUnit = (int) (Math.log10(nParts) / 3);
		System.out
				.println(String.format("Expect %d %s samples %d %s parts",
						(int) (nSamples / Math.pow(10, samplesUnit * 3)),
						UNITS[samplesUnit],
						(int) (nParts / Math.pow(10, partsUnit * 3)),
						UNITS[partsUnit]));
	}

	private static void compress() {
		final String[] idsList = DataHandler.getIdsPathList(parameters);
		final String[] nameList = DataHandler.getIdsNameList(parameters);
		List<String> codePathList = new ArrayList<String>();
		List<String> timing = new ArrayList<String>();
		List<String> logs = new ArrayList<String>();

		// single process executor
		ExecutorService executorService = Executors.newFixedThreadPool(1);
		List<Callable<double[][]>> compressorQueue = new ArrayList<Callable<double[][]>>();
		try {
			for (int i = 0; i < idsList.length; i++) {
				final int idsIdx = i;
				compressorQueue.add(new Callable<double[][]>() {

					private Compressor compressor;
					private TimerTask estimateSpeed = new TimerTask() {

						@Override
						public void run() {
							int processingSamples = processedSamples.get()
									+ compressor.countProcessedSamples();
							samplesCompressSpeed = (processingSamples - passedNSamples
									.get()) / (DELTA_TIME / 1000);
							passedNSamples.set(processingSamples);

							int processingParts = processedParts.get()
									+ compressor.countProcessedParts();
							partsCompressSpeed = (processingParts - passedNParts
									.get()) / (DELTA_TIME / 1000);
							passedNParts.set(processingParts);
						}
					};
					private TimerTask printState = new TimerTask() {

						@Override
						public void run() {
							System.out.println("Running " + " progress "
									+ compressor.countProcessedSamples() + "/"
									+ compressor.getNSamples() + " part "
									+ compressor.countProcessedParts() + "/"
									+ compressor.getNParts());
							System.out.println(" Speed " + samplesCompressSpeed
									+ " samples/sec " + partsCompressSpeed
									+ " parts/sec");
						}
					};

					@Override
					public double[][] call() throws Exception {
						float[] inputAudioData = DataHandler.audioread(
								idsList[idsIdx], parameters.getInExtension());
						compressor = new Compressor(inputAudioData, parameters);

						// start speed estimation
						Timer speedEstimator = new Timer();
						speedEstimator.scheduleAtFixedRate(estimateSpeed, 0,
								DELTA_TIME);

						// progress report
						Timer printTimer = null;
						if (parameters.getProgressReportRate() > 0) {
							printTimer = new Timer();
							printTimer.scheduleAtFixedRate(printState, 0,
									parameters.getProgressReportRate());
						}

						// process compressor
						double[][] codes = compressor.compress();
						processedSamples.addAndGet(compressor
								.countProcessedSamples());
						processedParts.addAndGet(compressor
								.countProcessedParts());
						speedEstimator.cancel();
						if (printTimer != null) {
							printTimer.cancel();
						}

						// logging
						timing.add(idsIdx,
								String.format("%d", compressor.time()));
						String log = "Compressed " + idsIdx + " "
								+ nameList[idsIdx] + " progress "
								+ compressor.countProcessedSamples() + "/"
								+ compressor.getNSamples() + " part "
								+ compressor.countProcessedParts() + "/"
								+ compressor.getNParts() + " time "
								+ compressor.time() / 1000 + " sec";
						logs.add(log);
						System.out.println(log);

						// store minimum value of self similarity
						Paths.get(parameters.getOutdir(), "\\",
								nameList[idsIdx]).getParent().toFile().mkdirs();
						String codeFilePath = Paths.get(parameters.getOutdir(),
								"\\", nameList[idsIdx]).toString();
						DataHandler.writecode(codeFilePath, codes,
								parameters.getOutExtension());
						codePathList.add(codeFilePath + "."
								+ parameters.getOutExtension());

						Files.write(Paths.get(parameters.getOutdir(),
								"\\codelist.txt"), codePathList);
						Files.write(Paths.get(parameters.getOutdir(),
								"\\timing.txt"), timing);
						Files.write(Paths.get(parameters.getOutdir(),
								"\\compresslog.txt"), logs);
						return codes;
					}
				});

			}
			executorService.invokeAll(compressorQueue);
		} catch (InterruptedException e) {
			e.printStackTrace();
		} finally {
			executorService.shutdown();
		}
	}

	public static void exec() {
		readParameters();
		System.out.println("Test name " + parameters.getTestName());
		System.out.println(parameters.toString());
		prepare();
		estimate();
		compress();
	}
}
