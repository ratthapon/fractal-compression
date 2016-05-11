package th.ac.kmitl.it.prip.fractal.compression.audio;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

import th.ac.kmitl.it.prip.fractal.DataHandler;
import th.ac.kmitl.it.prip.fractal.Executer;

public class CompressorExecuter extends Executer {
	private static final Logger LOGGER = Logger
			.getLogger(CompressorExecuter.class.getName());

	private static AtomicInteger processedSamples = new AtomicInteger(0);
	private static AtomicInteger processedParts = new AtomicInteger(0);
	private static AtomicInteger passedNSamples = new AtomicInteger(0);
	private static AtomicInteger passedNParts = new AtomicInteger(0);

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

					@Override
					public double[][] call() throws Exception {
						float[] inputAudioData = DataHandler.audioread(
								idsList[idsIdx], parameters.getInExtension());
						compressor = new Compressor(inputAudioData, parameters);

						// process compressor
						double[][] codes = compressor.compress();

						// logging
						writeLogs(compressor, idsIdx, nameList, timing, logs);
						// store minimum value of self similarity
						writeFractalCode(idsIdx, codes, nameList, codePathList);
						return codes;
					}
				});
			}
			executorService.invokeAll(compressorQueue);
		} catch (InterruptedException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
		} finally {
			executorService.shutdown();
		}
	}

	public static void exec() {
		readParameters();
		LOGGER.log(Level.INFO, "Test name " + parameters.getTestName());
		LOGGER.log(Level.INFO, parameters.toString());
		if (parameters.isValidParams()) {
			prepare();
			estimate();
			compress();
		}
	}

	private static void writeFractalCode(final int idsIdx, double[][] codes,
			final String[] nameList, List<String> codePathList)
			throws IOException {
		// store minimum value of self similarity
		Paths.get(parameters.getOutdir(), "\\", nameList[idsIdx]).getParent()
				.toFile().mkdirs();
		String codeFilePath = Paths.get(parameters.getOutdir(), "\\",
				nameList[idsIdx]).toString();
		DataHandler
				.writecode(codeFilePath, codes, parameters.getOutExtension());
		codePathList.add(codeFilePath + "." + parameters.getOutExtension());
		Files.write(Paths.get(parameters.getOutdir(), "\\codelist.txt"),
				codePathList);
	}

	private static void writeLogs(Compressor compressor, final int idsIdx,
			final String[] nameList, List<String> timing, List<String> logs)
			throws IOException {
		// logging
		timing.add(idsIdx, String.format("%d", compressor.time()));
		String log = "Compressed " + idsIdx + " " + nameList[idsIdx]
				+ " progress " + compressor.countProcessedSamples() + "/"
				+ compressor.getNSamples() + " part "
				+ compressor.countProcessedParts() + "/"
				+ compressor.getNParts() + " time " + compressor.time() / 1000
				+ " sec";
		logs.add(log);
		LOGGER.log(Level.INFO, log);
		Files.write(Paths.get(parameters.getOutdir(), "\\timing.txt"), timing);
		Files.write(Paths.get(parameters.getOutdir(), "\\compresslog.txt"),
				logs);
	}

	private CompressorExecuter() {

	}

	public static AtomicInteger getProcessedSamples() {
		return processedSamples;
	}

	public static AtomicInteger getProcessedParts() {
		return processedParts;
	}

	public static AtomicInteger getPassedNSamples() {
		return passedNSamples;
	}

	public static AtomicInteger getPassedNParts() {
		return passedNParts;
	}
}
