package th.ac.kmitl.it.prip.fractal.decompression.audio;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.sound.sampled.UnsupportedAudioFileException;

import th.ac.kmitl.it.prip.fractal.DataHandler;
import th.ac.kmitl.it.prip.fractal.Executer;
import th.ac.kmitl.it.prip.fractal.Parameters.ProcessName;

public class DecompressorExecuter extends Executer {
	private static final Logger LOGGER = Logger
			.getLogger(DecompressorExecuter.class.getName());


	private static void decompress() throws IOException {
		String[] idsList;
		try {
			idsList = DataHandler.getIdsPathList(parameters);
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, "Can not open file paths list : "
					+ parameters.getInPathPrefix());
			throw e;
		}
		String[] nameList;
		try {
			nameList = DataHandler.getIdsNameList(parameters);
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, "Can not open file names list : "
					+ parameters.getInfile());
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

	public static void exec() throws IOException, UnsupportedAudioFileException {
		try {
			readParameters();
			LOGGER.log(Level.INFO, "Test name " + parameters.getTestName());
			LOGGER.log(Level.INFO, parameters.toString());
			prepare();
			estimate(ProcessName.DECOMPRESS);
			decompress();
		} catch (IOException e) {
			LOGGER.info(e.getMessage());
			throw e;
		}
	}

	private DecompressorExecuter() {

	}

}
