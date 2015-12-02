package th.ac.kmitl.it.prip.fractal;

import th.ac.kmitl.it.prip.fractal.compression.audio.CompressorExecuter;

public class MainExecuter {

	public static void main(String[] args) {
		if (args.length > 0) {
			switch (args[0].toLowerCase()) {
			default:
			case "c":
			case "-c":
				CompressorExecuter.exec();
				break;
			case "d":
			case "-d":
				// DecompressorExecuter.exec();
				break;
			}

		} else {

		}

	}

}
