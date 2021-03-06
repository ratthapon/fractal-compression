package th.ac.kmitl.it.prip.fractal;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import javax.sound.sampled.UnsupportedAudioFileException;

import th.ac.kmitl.it.prip.fractal.compression.audio.ProcessDispatcher;

public class MainExecuter {
	private static final Logger LOGGER = Logger.getLogger(MainExecuter.class.getName());

	private MainExecuter() {
	}

	protected static List<String> readInput() throws IOException {
		List<String> parametersList = new ArrayList<>();
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
			String parameter = br.readLine();
			while (parameter != null && parameter.length() > 0) {
				parametersList.add(parameter);
				parameter = br.readLine();
			}
			br.close();
		} catch (IOException e) {
			LOGGER.info(e.getMessage());
			throw e;
		}
		return parametersList;
	}

	public static void main(String[] args) throws IOException, UnsupportedAudioFileException, InterruptedException {
		try {
			if (args.length > 0) {
				System.setIn(new FileInputStream(args[0]));
			}
			List<String> argParams = readInput();
			String[] paramBuffer = new String[argParams.size()];
			argParams.toArray(paramBuffer);
			Parameters parameters = new Parameters(paramBuffer);
			ProcessDispatcher dispatcher = new ProcessDispatcher(parameters);
			dispatcher.exec();
		} catch (IOException e) {
			LOGGER.info(e.getMessage());
			throw e;
		}
	}
}
