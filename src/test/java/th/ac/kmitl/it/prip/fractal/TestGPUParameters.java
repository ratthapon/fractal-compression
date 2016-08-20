package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public class TestGPUParameters {
	private String inputParam;
	private boolean expected;

	@Parameterized.Parameters
	public static Collection<Object[]> setGPUEnable() {
		return Arrays.asList(new Object[][] { { "gpu true", true }, { "gpu false", false }, { "gpu True", true },
				{ "GPU false", false }, });
	}

	public TestGPUParameters(String inputParam, boolean expectedResult) {
		this.inputParam = inputParam;
		this.expected = expectedResult;
	}

	@Test
	public void correctSetting() throws IOException {
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expected, parameters.isGpuEnable());
	}

}
