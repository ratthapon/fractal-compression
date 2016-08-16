package th.ac.kmitl.it.prip.fractal;

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
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.sound.sampled.UnsupportedAudioFileException;

import th.ac.kmitl.it.prip.fractal.Parameters;
import th.ac.kmitl.it.prip.fractal.Parameters.ProcessName;
import th.ac.kmitl.it.prip.fractal.compression.audio.Compressor;
import th.ac.kmitl.it.prip.fractal.decompression.audio.Decompressor;

public abstract class Executer {
	private static final Logger LOGGER = Logger.getLogger(Executer.class
			.getName());

	protected static final String[] UNITS = { "", "k", "M", "G", "T", "P" };
	public static final int DELTA_TIME = 5000;
	protected static Parameters parameters;
	protected static String[] inputParams;
	protected static int nSamples = 0;
	protected static int nParts = 0;

	protected static void processParameters(String[] lines) throws IOException {
		inputParams = lines;
		processParameters();
	}

	public static void processParameters(List<String> lines) throws IOException {
		String[] result = new String[lines.size()];
		for (int i = 0; i < lines.size(); i++) {
			result[i] = lines.get(i);
		}
		inputParams = result;
		processParameters();
	}

	protected static void processParameters() throws IOException {
		parameters = new Parameters(inputParams);
		if (parameters.isHelp()) {
			try {
				InputStream in = MainExecuter.class
						.getResourceAsStream("help.txt");
				BufferedReader br = new BufferedReader(
						new InputStreamReader(in));
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

	@SuppressWarnings("unused")
	private static void clean() {
		// remove output directory
		// remove parameter setting
		// remove output
	}

	protected static void prepare() throws IOException {
		// generate output directory
		Paths.get(parameters.getOutdir()).toFile().mkdirs();

		// generate parameter setting
		List<String> info = new ArrayList<String>();
		// save process description
		try {
			info.add("Java " + System.getProperty("java.version"));
			info.add("Version "
					+ Executors.class.getPackage().getImplementationVersion());
			info.add("Date " + LocalDateTime.now());
			info.add(parameters.toString());

			Files.write(Paths.get(parameters.getOutdir(), "\\info.txt"), info);
			Files.write(Paths.get(parameters.getOutdir(), "\\parameters.txt"),
					Arrays.asList(inputParams));
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}

		// generate output
	}

	protected static void estimate(ProcessName processName) throws IOException,
			UnsupportedAudioFileException {
		// estimate runtime
		String[] idsList = null;
		try {
			idsList = DataHandler.getIdsPathList(parameters);
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE,
					"Can not open files list : " + parameters.getInfile());
			throw e;
		}
		for (int i = 0; i < idsList.length; i++) {
			try {
				if (processName.equals(ProcessName.COMPRESS)) {
					float[] inputAudioData = DataHandler.audioread(idsList[i],
							parameters.getInExtension());
					Compressor compressor = new Compressor(inputAudioData,
							parameters);
					nSamples += compressor.getNSamples();
					nParts += compressor.getNParts();
				} else if (processName.equals(ProcessName.DECOMPRESS)) {
					double[][] codes = DataHandler.codesread(idsList[i],
							parameters.getInExtension());
					Decompressor decompressor = new Decompressor(codes,
							parameters);
					nSamples += decompressor.getNSamples();
					nParts += decompressor.getNParts();
				}

			} catch (UnsupportedAudioFileException e) {
				LOGGER.log(Level.SEVERE, "Unsupport audio file extension : "
						+ parameters.getInExtension());
				throw e;
			} catch (IOException e) {
				LOGGER.log(Level.SEVERE, "File read error : " + idsList[i]);
				throw e;
			}
		}
		int samplesUnit = (int) (Math.log10(nSamples) / 3);
		int partsUnit = (int) (Math.log10(nParts) / 3);
		LOGGER.log(Level.INFO, String.format(
				"Expect %d %s samples %d %s parts",
				(int) (nSamples / Math.pow(10, samplesUnit * 3)),
				UNITS[samplesUnit],
				(int) (nParts / Math.pow(10, partsUnit * 3)), UNITS[partsUnit]));
	}

	protected Executer() {
	}
}