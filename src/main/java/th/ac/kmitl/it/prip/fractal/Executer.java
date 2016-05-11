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

public abstract class Executer {
	private static final Logger LOGGER = Logger.getLogger(Executer.class
			.getName());

	protected static final String[] UNITS = { "", "k", "M", "G", "T", "P" };
	protected static final int DELTA_TIME = 5000;
	protected static Parameters parameters;
	private static String[] inputParams;
	protected static int nSamples = 0;
	protected static int nParts = 0;

	protected static void readParameters() {
		List<String> parametersList = new ArrayList<>();
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(
					System.in));
			String parameter = br.readLine();
			while (parameter.length() > 0) {
				parametersList.add(parameter);
				parameter = br.readLine();
			}
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
		}
		String[] result = new String[parametersList.size()];
		for (int i = 0; i < parametersList.size(); i++) {
			result[i] = parametersList.get(i);
		}
		inputParams = result;
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

	protected static void prepare() {
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
		}

		// generate output
	}

	protected Executer() {
	}
}