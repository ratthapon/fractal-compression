package th.ac.kmitl.it.prip.fractal.dataset;

import java.io.InputStream;

interface DataSetAPIv1 {
	public float[] readAudio(int index);

	public boolean writeAudo(int index);

	public double[][] readCode(int index);

	public boolean writeCode(int index);

	public boolean updateTimingLog(int index);

	public boolean updateCompletedLog(int index);

	public boolean writeInfo(int index);

	public boolean writeParameters(int index);

	public boolean writeCodePaths();

	public int estimateNumSamples();

	public DataSetManager setParameters(InputStream is);

}
