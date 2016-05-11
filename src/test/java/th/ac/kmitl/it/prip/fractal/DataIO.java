package th.ac.kmitl.it.prip.fractal;

import org.junit.Test;

public class DataIO {
	@Test
	public void inRAW_TO_outRAW() {
		String fileName = "F://IFEFSR//SpeechData//an4//wav//an4_clstk//fash//an251-fash-b";
		String outPath = "F://IFEFSR//SpeechData//an4//wav//an251-fash-b";
		double[] outRaw;
		float[] inRaw;
		inRaw = DataHandler.audioread(fileName, "raw");
		outRaw = new double[inRaw.length];
		for (int i = 0; i < inRaw.length; i++) {
			outRaw[i] = (double) inRaw[i];
		}
		DataHandler.writeaudio(outPath, outRaw, "raw");
	}
}
