package th.ac.kmitl.it.prip.fractal;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public abstract class Executer {
	private static final Logger LOGGER = Logger.getLogger(Executer.class.getName());

	
	public static final int DELTA_TIME = 5000;
	protected static Parameters parameters;
	protected static String[] inputParams;

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
				InputStream in = MainExecuter.class.getResourceAsStream("help.txt");
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

	@SuppressWarnings("unused")
	private static void clean() {
		// remove output directory
		// remove parameter setting
		// remove output
	}

	protected Executer() {
	}
}