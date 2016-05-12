package th.ac.kmitl.it.prip.fractal;

import th.ac.kmitl.it.prip.fractal.compression.audio.CompressorExecuter;

public class MainExecuter {

	public static void main(String[] args) {
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
	}

	private MainExecuter() {
	}

}
