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

	protected static void process() throws IOException, InterruptedException {
		String[] idsList;
		String[] nameList;
		try {
			idsList = DataHandler.getIdsPathList(parameters);
			nameList = DataHandler.getIdsNameList(parameters);
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE,
					"Can not open file list : " + parameters.getInPathPrefix());
			throw e;
		}

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
			throw e;
		} finally {
			executorService.shutdown();
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
