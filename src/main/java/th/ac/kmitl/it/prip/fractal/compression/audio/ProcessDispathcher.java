package th.ac.kmitl.it.prip.fractal.compression.audio;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.sound.sampled.UnsupportedAudioFileException;

import th.ac.kmitl.it.prip.fractal.Executer;
import th.ac.kmitl.it.prip.fractal.compression.audio.gpu.CUCompressor;
import th.ac.kmitl.it.prip.fractal.dataset.DataSetManager;

public class CompressorExecuter extends Executer {
	private static final Logger LOGGER = Logger.getLogger(CompressorExecuter.class.getName());

	private static AtomicInteger processedSamples = new AtomicInteger(0);
	private static AtomicInteger processedParts = new AtomicInteger(0);
	private static AtomicInteger passedNSamples = new AtomicInteger(0);
	private static AtomicInteger passedNParts = new AtomicInteger(0);

	protected static void process() throws IOException, InterruptedException {
		String[] idsList;
		String[] nameList;
		try {
			idsList = DataSetManager.getIdsPathList(parameters);
			nameList = DataSetManager.getIdsNameList(parameters);
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, "Can not open file list : " + parameters.getInPathPrefix());
			throw e;
		}

		List<String> codePathList = new ArrayList<String>(idsList.length);
		List<String> timing = new ArrayList<String>(idsList.length);
		List<String> logs = new ArrayList<String>(idsList.length);

		// single process executor
		ExecutorService executorService = Executors.newFixedThreadPool(1);
		List<Callable<double[][]>> compressorQueue = new ArrayList<Callable<double[][]>>();

		for (int i = 0; i < idsList.length; i++) {
			final int idsIdx = i;
			codePathList.add("");
			timing.add("");
			logs.add("");
			Path codeFilePath = Paths.get(parameters.getOutdir(), "\\",
					nameList[idsIdx] + "." + parameters.getOutExtension());
			File audio = new File(codeFilePath.toString());
			if (audio.exists() && parameters.isSkipIfExist()) {
				writeSkipLogs(idsIdx, nameList, timing, logs);
			} else {
				compressorQueue.add(new Callable<double[][]>() {

					@Override
					public double[][] call() throws Exception {
						double[][] codes = null;
						float[] inputAudioData = DataSetManager.audioread(idsList[idsIdx], parameters.getInExtension());
						if (parameters.isGpuEnable()) {
							// process compressor
							CUCompressor compressor = new CUCompressor(inputAudioData, parameters);
							codes = compressor.compress();
							// logging
							writeLogs(compressor, idsIdx, nameList, timing, logs);
						} else {
							Compressor compressor = new Compressor(inputAudioData, parameters);
							codes = compressor.compress();
							// logging
							writeLogs(compressor, idsIdx, nameList, timing, logs);
						}

						// store minimum value of self similarity
						writeFractalCode(idsIdx, codes, nameList, codePathList);
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

	public static void exec() throws IOException, UnsupportedAudioFileException, InterruptedException {
		try {
			processParameters();
			LOGGER.log(Level.INFO, "Test name " + parameters.getTestName());
			LOGGER.log(Level.INFO, parameters.toString());
			if (parameters.isValidParams()) {
				prepare();
				estimate(parameters.getProcessName());
				process();
			}
		} catch (IOException | UnsupportedAudioFileException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
	}

	private static void writeFractalCode(final int idsIdx, double[][] codes, final String[] nameList,
			List<String> codePathList) throws IOException {
		// store minimum value of self similarity
		Paths.get(parameters.getOutdir(), "\\", nameList[idsIdx]).getParent().toFile().mkdirs();
		String codeFilePath = Paths.get(parameters.getOutdir(), "\\", nameList[idsIdx]).toString();
		DataSetManager.writecode(codeFilePath, codes, parameters.getOutExtension());
		codePathList.set(idsIdx, codeFilePath + "." + parameters.getOutExtension());
		Files.write(Paths.get(parameters.getOutdir(), "\\codelist.txt"), codePathList);
	}

	private static void writeLogs(Compressor compressor, final int idsIdx, final String[] nameList, List<String> timing,
			List<String> logs) throws IOException {
		// logging
		timing.set(idsIdx, String.format("%d", compressor.time()));
		String log = "Compressed " + idsIdx + " " + nameList[idsIdx] + " progress " + compressor.countProcessedSamples()
				+ "/" + compressor.getNSamples() + " part " + compressor.countProcessedParts() + "/"
				+ compressor.getNParts() + " time " + compressor.time() / 1000 + " sec";
		logs.set(idsIdx, log);
		LOGGER.log(Level.INFO, log);
		Files.write(Paths.get(parameters.getOutdir(), "\\timing.txt"), timing);
		Files.write(Paths.get(parameters.getOutdir(), "\\compresslog.txt"), logs);
	}

	private static void writeSkipLogs(final int idsIdx, final String[] nameList, List<String> timing, List<String> logs)
			throws IOException {
		// logging
		timing.set(idsIdx, String.format("%d", 0));
		String log = "Skiped " + idsIdx + " " + nameList[idsIdx];
		logs.set(idsIdx, log);
		LOGGER.log(Level.INFO, log);
		Files.write(Paths.get(parameters.getOutdir(), "\\timing.txt"), timing);
		Files.write(Paths.get(parameters.getOutdir(), "\\compresslog.txt"), logs);
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

	private static Compressor getCompressor(float[] inputAudioData) {
		if (parameters.isGpuEnable()) {
			return new CUCompressor(inputAudioData, parameters);
		} else {
			return new Compressor(inputAudioData, parameters);
		}
	}
}
