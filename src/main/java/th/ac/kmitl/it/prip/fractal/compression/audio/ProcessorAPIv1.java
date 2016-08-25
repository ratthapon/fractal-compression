package th.ac.kmitl.it.prip.fractal.compression.audio;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;

public interface ProcessorAPIv1 {
	public double[][] process() throws InterruptedException, ExecutionException;
	
	public void registerPartCount(AtomicInteger counter);
	public void registerSampleCount(AtomicInteger counter);

}
