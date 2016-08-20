package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public class ParametersIntTest {
	private int input;
	private int expected;

	@Parameterized.Parameters
	public static Collection<Object[]> setIntParameters() {
		return Arrays.asList(new Object[][] { { 1, 1 }, { 2, 2 }, { 5, 5 }, { 1000, 1000 }, { -10000, -10000 } });
	}

	public ParametersIntTest(int input, int expected) {
		this.input = input;
		this.expected = expected;
	}

	@Test
	public void testMaxParallelProcess() throws IOException {
		String inputParam = String.format("maxprocess %d", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.getMaxParallelProcess());
	}

	@Test
	public void testVerbose() throws IOException {
		String inputParam = String.format("reportrate %d", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.getProgressReportRate());
	}

	@Test
	public void tesetOverlap() throws IOException {
		String inputParam = String.format("overlap %d", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.getOverlap());
	}

	@Test
	public void testDStep() throws IOException {
		String inputParam = String.format("step %d", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.getDStep());
	}

	@Test
	public void testNCoeff() throws IOException {
		String inputParam = String.format("ncoeff %d", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.getNCoeff());
	}

	@Test
	public void testMinRBS() throws IOException {
		String inputParam = String.format("minr %d", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.getMinBlockSize());
	}

	@Test
	public void testMaxRBS() throws IOException {
		String inputParam = String.format("maxr %d", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.getMaxBlockSize());
	}

	@Test
	public void testDScale() throws IOException {
		String inputParam = String.format("AA %d", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.getDomainScale());
	}

	@Test
	public void testFrameLength() throws IOException {
		String inputParam = String.format("framelength %d", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.getFrameLength());
	}

	@Test
	public void testMaxIteration() throws IOException {
		String inputParam = String.format("maxiter %d", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname decompress" });
		assertEquals(this.expected, parameters.getMaxIteration());
	}

	@Test
	public void testSamplingRate() throws IOException {
		String inputParam = String.format("fs %d", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname decompress" });
		assertEquals(this.expected, parameters.getSamplingRate());
	}

}
