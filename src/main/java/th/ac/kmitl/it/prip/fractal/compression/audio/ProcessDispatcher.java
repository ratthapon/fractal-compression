package th.ac.kmitl.it.prip.fractal.compression.audio;

import java.io.IOException;
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
import th.ac.kmitl.it.prip.fractal.Parameters;
import th.ac.kmitl.it.prip.fractal.compression.audio.gpu.CUCompressor;
import th.ac.kmitl.it.prip.fractal.dataset.DataSetManager;

public class ProcessDispatcher extends Executer {
	private static final Logger LOGGER = Logger.getLogger(ProcessDispatcher.class.getName());
	protected static final String[] UNITS = { "", "k", "M", "G", "T", "P" };

	private static AtomicInteger processedSamples = new AtomicInteger(0);
	private static AtomicInteger processedParts = new AtomicInteger(0);
	private static AtomicInteger passedNSamples = new AtomicInteger(0);
	private static AtomicInteger passedNParts = new AtomicInteger(0);
	private DataSetManager dataSetManager;

	protected void dispatch() throws IOException, InterruptedException {
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
							codes = compressor.compress();
							// logging

							writeLogs(compressor, idsIdx);
						} else {
							Compressor compressor = new Compressor(inputAudioData, parameters);
							codes = compressor.compress();
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

	protected void estimate() throws IOException, UnsupportedAudioFileException {
		int nSamples = dataSetManager.estimateNumSamples();
		int samplesUnit = (int) (Math.log10(nSamples) / 3);
		LOGGER.log(Level.INFO, String.format("Expect %d %s samples", (int) (nSamples / Math.pow(10, samplesUnit * 3)),
				UNITS[samplesUnit]));
	}

	public void exec() throws IOException, UnsupportedAudioFileException, InterruptedException {
		try {
			processParameters();
			LOGGER.log(Level.INFO, "Test name " + parameters.getTestName());
			LOGGER.log(Level.INFO, parameters.toString());
			if (parameters.isValidParams()) {
				prepare();
				estimate();
				dispatch();
			}
		} catch (IOException | UnsupportedAudioFileException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
	}

	private void writeLogs(Compressor compressor, final int idsIdx) throws IOException {
		// logging
		String log = "Compressed " + idsIdx + " " + dataSetManager.getName(idsIdx) + " progress "
				+ compressor.countProcessedSamples() + "/" + compressor.getNSamples() + " part "
				+ compressor.countProcessedParts() + "/" + compressor.getNParts() + " time " + compressor.time() / 1000
				+ " sec";
		LOGGER.log(Level.INFO, log);
		dataSetManager.updateTimingLog(idsIdx, compressor.time());
		dataSetManager.updateCompletedLog(idsIdx, log);
	}

	public ProcessDispatcher(Parameters parameters) throws IOException {
		this.dataSetManager = new DataSetManager(parameters);
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
