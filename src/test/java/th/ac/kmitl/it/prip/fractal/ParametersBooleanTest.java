package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public class ParametersBooleanTest {
	private String input;
	private boolean expected;

	@Parameterized.Parameters
	public static Collection<Object[]> setBooleanParameters() {
		return Arrays
				.asList(new Object[][] { { "true", true }, { "false", false }, { "True", true }, { "false", false }, });
	}

	public ParametersBooleanTest(String input, boolean expectedResult) {
		this.input = input;
		this.expected = expectedResult;
	}

	@Test
	public void testSkipIfExist() throws IOException {
		String inputParam = String.format("skipifexist %s", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.isSkipIfExist());
	}

	@Test
	public void testADP() throws IOException {
		String inputParam = String.format("adaptive %s", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.isAdaptivePartition());
	}

	@Test
	public void testUsingCV() throws IOException {
		String inputParam = String.format("cv %s", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.isUsingCV());
	}

	@Test
	public void testGPU() throws IOException {
		String inputParam = String.format("gpu %s", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.isGpuEnable());
	}

}
