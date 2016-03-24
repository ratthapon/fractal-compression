package th.ac.kmitl.it.prip.fractal.decompression.audio;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import th.ac.kmitl.it.prip.fractal.DataHandler;
import th.ac.kmitl.it.prip.fractal.Executer;

public class DecompressorExecuter extends Executer {
	private static void estimate() {
		// estimate runtime
		final String[] idsList = DataHandler.getIdsPathList(parameters);
		for (int i = 0; i < idsList.length && i < parameters.getToIdx(); i++) {
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

		for (int idsIdx = parameters.getFromIdx(); idsIdx < idsList.length
				&& idsIdx < parameters.getToIdx(); idsIdx++) {
			Decompressor decompressor;
			Paths.get(parameters.getOutdir(), "\\", nameList[idsIdx])
					.getParent().toFile().mkdirs();
			File audioFile = Paths.get(parameters.getOutdir(), "\\",
					nameList[idsIdx]).toFile();
			String audioFilePath = audioFile.toString();
			audioPathList.add(audioFilePath + "."
					+ parameters.getOutExtension());
			double[][] codes = DataHandler.codesread(idsList[idsIdx],
					parameters.getInExtension());
			decompressor = new Decompressor(codes, parameters);
			double[] audioData = null;
			if (parameters.isSkipIfExist() && audioFile.exists()) {
				// skip
			} else {
				audioData = decompressor.decompress();
				// store decoded audio
				DataHandler.writeaudio(audioFilePath, audioData, parameters);
			}
			// logging
			timing.add(idsIdx, String.format("%d", decompressor.time()));
			String log = "Decompressed " + idsIdx + " " + nameList[idsIdx]
					+ " time " + decompressor.time() / 1000 + " sec";
			logs.add(log);
			System.out.println(log);

			try {
				Files.write(
						Paths.get(parameters.getOutdir(), "\\audiolist.txt"),
						audioPathList);
				Files.write(Paths.get(parameters.getOutdir(),
						"\\decompresstiming.txt"), timing);
				Files.write(Paths.get(parameters.getOutdir(),
						"\\decompresslog.txt"), logs);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	public static void exec() {
		prepare();
		estimate();
		decompress();
		System.out.println("Complete Exec");
	}

}
