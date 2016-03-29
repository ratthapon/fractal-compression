package th.ac.kmitl.it.prip.fractal;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import th.ac.kmitl.it.prip.fractal.compression.audio.CompressorExecuter;
import th.ac.kmitl.it.prip.fractal.decompression.audio.DecompressorExecuter;

public class MainExecuter {
	protected static List<String> readInput() {
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
			e.printStackTrace();
		}
		return parametersList;
	}

	public static void exec() {
		switch (Executer.parameters.getProcessName()) {
		default:
		case COMPRESS:
			CompressorExecuter.exec();
			break;
		case DECOMPRESS:
			DecompressorExecuter.exec();
			break;
		}
	}

	public static void main(String[] args) {
		try {
			if (args.length > 0) {
				System.setIn(new FileInputStream(args[0]));
			}
			Executer.processParameters(readInput());
			exec();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
