package th.ac.kmitl.it.prip.fractal.dataset;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

import javax.sound.sampled.UnsupportedAudioFileException;

interface DataSetAPIv1 {
	public float[] readAudio(int index) throws IOException, UnsupportedAudioFileException;

	public boolean writeAudio(int index, double[] audioData) throws IOException;

	public double[][] readCode(int index) throws IOException;

	public boolean writeCode(int index, double[][] codeData) throws Exception;

	public boolean updateTimingLog(int index, int time) throws IOException;

	public boolean updateCompletedLog(int index, String logString) throws IOException;

	public boolean writeInfo() throws IOException;

	public boolean writeParameters() throws IOException;

	public boolean writeOutputPaths() throws IOException;

	public int estimateNumSamples() throws IOException;

}
