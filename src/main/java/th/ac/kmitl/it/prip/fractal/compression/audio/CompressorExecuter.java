package th.ac.kmitl.it.prip.fractal.compression.audio;

import java.io.File;
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
import th.ac.kmitl.it.prip.fractal.compression.audio.gpu.CUCompressor;

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
		for (int i = parameters.getFromIdx(); i < idsList.length
				&& i < parameters.getToIdx(); i++) {
			float[] inputAudioData = DataHandler.audioread(idsList[i],
					parameters.getInExtension());
			Compressor compressor = new Compressor(inputAudioData, parameters);
			nSamples += compressor.getNSamples();
			nParts += compressor.getNParts();
		}
		int samplesUnit = (int) (Math.log10(nSamples) / 3);
		int partsUnit = (int) (Math.log10(nParts) / 3);
		System.out.println(String.format(
				"Expect %.2f %s samples %.2f %s parts",
				(nSamples / Math.pow(10, samplesUnit * 3)), UNITS[samplesUnit],
				(nParts / Math.pow(10, partsUnit * 3)), UNITS[partsUnit]));
	}

	private static void compress() {
		final String[] idsList = DataHandler.getIdsPathList(parameters);
		final String[] nameList = DataHandler.getIdsNameList(parameters);
		List<String> codePathList = new ArrayList<String>(idsList.length);
		List<String> timing = new ArrayList<String>(idsList.length);
		List<String> logs = new ArrayList<String>(idsList.length);
		for (int i = 0; i < idsList.length; i++) {
			codePathList.add(new String());
			timing.add(new String());
			logs.add(new String());
		}

		// single process executor
		ExecutorService executorService = Executors.newFixedThreadPool(1);
		List<Callable<double[][]>> compressorQueue = new ArrayList<Callable<double[][]>>();
		try {
			for (int i = parameters.getFromIdx(); i < idsList.length
					&& i < parameters.getToIdx(); i++) {
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
						// retrieve information
						File codeFile = Paths.get(parameters.getOutdir(), "\\",
								nameList[idsIdx]).toFile();
						String codeFilePath = codeFile.toString();
						codeFile = Paths.get(
								parameters.getOutdir(),
								"\\",
								nameList[idsIdx] + "."
										+ parameters.getOutExtension())
								.toFile();
						float[] inputAudioData = DataHandler.audioread(
								idsList[idsIdx], parameters.getInExtension());
						codePathList.set(idsIdx, nameList[idsIdx] + "."
								+ parameters.getOutExtension());
						try {
							if (parameters.isGpuEnable()) {
								compressor = new CUCompressor(inputAudioData,
										parameters);
							} else {
								compressor = new Compressor(inputAudioData,
										parameters);
							}
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
						double[][] codes = null;

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
						if (parameters.isSkipIfExist() && codeFile.exists()) {
							// skip
							timing.set(idsIdx, String.format("%d", 0));
							String log = "Skiped " + idsIdx + " "
									+ nameList[idsIdx] + " progress "
									+ compressor.getNSamples() + "/"
									+ compressor.getNSamples() + " part "
									+ compressor.getNParts() + "/"
									+ compressor.getNParts() + " time "
									+ compressor.time() / 1000 + " sec";
							logs.set(idsIdx, log);
							System.out.println(log);

						} else {
							if (parameters.isGpuEnable()) {
								codes = ((CUCompressor) compressor).compress();
							} else {
								codes = compressor.compress();
							}
							// logging
							timing.set(idsIdx,
									String.format("%d", compressor.time()));
							String log = "Compressed " + idsIdx + " "
									+ nameList[idsIdx] + " progress "
									+ compressor.countProcessedSamples() + "/"
									+ compressor.getNSamples() + " part "
									+ compressor.countProcessedParts() + "/"
									+ compressor.getNParts() + " time "
									+ compressor.time() / 1000 + " sec";
							logs.set(idsIdx, log);
							System.out.println(log);

							// store minimum value of self similarity
							Paths.get(parameters.getOutdir(), "\\",
									nameList[idsIdx]).getParent().toFile()
									.mkdirs();
							DataHandler.writecode(codeFilePath, codes,
									parameters.getOutExtension());
						}

						// cancel estimation
						processedSamples.addAndGet(compressor
								.countProcessedSamples());
						processedParts.addAndGet(compressor
								.countProcessedParts());
						speedEstimator.cancel();
						if (printTimer != null) {
							printTimer.cancel();
						}

						// store logs
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
		prepare();
		estimate();
		compress();
		System.out.println("Complete Exec");
	}
}
