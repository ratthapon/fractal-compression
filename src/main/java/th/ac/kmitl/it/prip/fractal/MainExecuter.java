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
import th.ac.kmitl.it.prip.fractal.decompression.audio.DecompressorExecuter;

public class MainExecuter {
	private static final Logger LOGGER = Logger.getLogger(MainExecuter.class.getName());

	private MainExecuter() {
	}

	protected static List<String> readInput() throws IOException {
		List<String> parametersList = new ArrayList<>();
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
			String parameter = br.readLine();
			while (parameter.length() > 0) {
				parametersList.add(parameter);
				parameter = br.readLine();
			}
		} catch (IOException e) {
			LOGGER.info(e.getMessage());
			throw e;
		}
		return parametersList;
	}

	public static void exec() throws IOException, UnsupportedAudioFileException, InterruptedException {
		switch (Executer.parameters.getProcessName()) {
		case DECOMPRESS:
			DecompressorExecuter.exec();
			break;
		case COMPRESS:
		default:
			ProcessDispatcher dispatcher = new ProcessDispatcher(Executer.parameters);
			dispatcher.exec();
			break;
		}
	}

	public static void main(String[] args) throws IOException, UnsupportedAudioFileException, InterruptedException {
		try {
			if (args.length > 0) {
				System.setIn(new FileInputStream(args[0]));
			}
			Executer.processParameters(readInput());
			exec();
		} catch (IOException e) {
			LOGGER.info(e.getMessage());
			throw e;
		}
	}
}
