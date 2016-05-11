package th.ac.kmitl.it.prip.fractal.decompression.audio;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import th.ac.kmitl.it.prip.fractal.DataHandler;
import th.ac.kmitl.it.prip.fractal.Executer;

public class DecompressorExecuter extends Executer {
	private static final Logger LOGGER = Logger
			.getLogger(DecompressorExecuter.class.getName());

	private static void estimate() {
		// estimate runtime
		final String[] idsList = DataHandler.getIdsPathList(parameters);
		for (int i = 0; i < idsList.length; i++) {
			double[][] codes = DataHandler.codesread(idsList[i],
					parameters.getInExtension());
			Decompressor decompressor = new Decompressor(codes, parameters);
			nSamples += decompressor.getNSamples();
			nParts += decompressor.getNParts();
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

	private static void decompress() {
		final String[] idsList = DataHandler.getIdsPathList(parameters);
		final String[] nameList = DataHandler.getIdsNameList(parameters);
		List<String> audioPathList = new ArrayList<String>();
		List<String> timing = new ArrayList<String>();
		List<String> logs = new ArrayList<String>();

		// single process executor

		for (int idsIdx = 0; idsIdx < idsList.length; idsIdx++) {
			Decompressor decompressor;
			LOGGER.log(Level.FINE, "Encall");
			double[][] codes = DataHandler.codesread(idsList[idsIdx],
					parameters.getInExtension());
			decompressor = new Decompressor(codes, parameters);
			LOGGER.log(Level.FINE, "Create decompressor");
			double[] audioData = decompressor.decompress();
			LOGGER.log(Level.FINE, "Finish decompress");
			// logging
			timing.add(idsIdx, String.format("%d", decompressor.time()));
			String log = "Decompressed " + idsIdx + " " + nameList[idsIdx]
					+ " time " + decompressor.time() / 1000 + " sec";
			logs.add(log);
			LOGGER.log(Level.INFO, log);

			// store minimum value of self similarity
			Paths.get(parameters.getOutdir(), "\\", nameList[idsIdx])
					.getParent().toFile().mkdirs();
			String audioFilePath = Paths.get(parameters.getOutdir(), "\\",
					nameList[idsIdx]).toString();
			DataHandler.writeaudio(audioFilePath, audioData,
					parameters.getOutExtension());
			audioPathList.add(audioFilePath + "."
					+ parameters.getOutExtension());
			try {
				Files.write(
						Paths.get(parameters.getOutdir(), "\\audiolist.txt"),
						audioPathList);
				Files.write(Paths.get(parameters.getOutdir(),
						"\\decompresstiming.txt"), timing);
				Files.write(Paths.get(parameters.getOutdir(),
						"\\decompresslog.txt"), logs);
			} catch (IOException e) {
				LOGGER.log(Level.SEVERE, e.getMessage());
			}
		}
		LOGGER.log(Level.FINE, "Complete Exec");
	}

	public static void exec() {
		readParameters();
		LOGGER.log(Level.INFO, "Test name " + parameters.getTestName());
		LOGGER.log(Level.INFO, parameters.toString());
		prepare();
		estimate();
		decompress();

	}

	private DecompressorExecuter() {

	}

}
