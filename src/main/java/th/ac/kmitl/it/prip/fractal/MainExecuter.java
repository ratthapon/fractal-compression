package th.ac.kmitl.it.prip.fractal;

import java.io.IOException;
import java.util.logging.Logger;

import javax.sound.sampled.UnsupportedAudioFileException;

import th.ac.kmitl.it.prip.fractal.compression.audio.CompressorExecuter;

public class MainExecuter {
	private static final Logger LOGGER = Logger.getLogger(MainExecuter.class
			.getName());

	public static void main(String[] args) throws IOException,
			UnsupportedAudioFileException, InterruptedException {
		try {
			if (args.length > 0) {
				switch (args[0].toLowerCase()) {
				case "c":
				case "-c":
				default:
					CompressorExecuter.exec();
					break;
				}

			} else {
				CompressorExecuter.exec();
			}
		} catch (IOException | UnsupportedAudioFileException
				| InterruptedException e) {
			LOGGER.info(e.getMessage());
			throw e;
		}
	}

	private MainExecuter() {
	}

}
