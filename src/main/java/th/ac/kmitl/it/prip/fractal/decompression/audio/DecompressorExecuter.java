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

	protected static void process() throws IOException {
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
				throw e;
			}
		}
		LOGGER.log(Level.FINE, "Complete Exec");
	}

	private DecompressorExecuter() {
	}

}
