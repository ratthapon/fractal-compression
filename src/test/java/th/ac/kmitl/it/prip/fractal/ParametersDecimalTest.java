package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public class ParametersDecimalTest {
	private float input;
	private float expected;
	private final float EPSILON = 1e-11f;

	@Parameterized.Parameters
	public static Collection<Object[]> setIntParameters() {
		return Arrays.asList(new Object[][] { { 1f, 1f }, { 1e-1f, 1e-1f }, { 5e-2f, 5e-2f }, { 1e-5f, 1e-5f },
				{ -1e-6f, -1e-6f } });
	}

	public ParametersDecimalTest(float input, float expected) {
		this.input = input;
		this.expected = expected;
	}

	@Test
	public void testCoeffLimit() throws IOException {
		String inputParam = String.format("coefflimit %f", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.getCoeffLimit(), EPSILON);
	}

	@Test
	public void testThresh() throws IOException {
		String inputParam = String.format("pthresh %f", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.getThresh(), EPSILON);
	}

	@Test
	public void testRegularize() throws IOException {
		String inputParam = String.format("regularize %f", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.getRegularize(), EPSILON);
	}

	@Test
	public void testAlpha() throws IOException {
		String inputParam = String.format("alpha %f", input);
		Parameters parameters = new Parameters(new String[] { inputParam, "processname decompress" });
		assertEquals(this.expected, parameters.getAlpha(), EPSILON);
	}

}
