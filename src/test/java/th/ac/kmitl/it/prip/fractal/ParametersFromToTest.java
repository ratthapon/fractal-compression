package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
public class ParametersFromToTest {
	private String inputParam;
	private int expectedFrom;
	private int expectedTo;

	@Parameterized.Parameters
	public static Collection<Object[]> setFromTo() {
		return Arrays.asList(new Object[][] { { "fromto 1-1", 1, 1 }, { "fromto 1-5", 1, 5 }, { "fromto 5-13", 5, 13 },
				{ "fromto 5-1", 5, 1 }, { "fromto 1 1", 0, -1 } });
	}

	public ParametersFromToTest(String inputParam, int expectedFrom, int expectedTo) {
		this.inputParam = inputParam;
		this.expectedFrom = expectedFrom - 1;
		this.expectedTo = expectedTo;
	}

	@Test
	public void correctSetting() throws IOException {
		Parameters parameters = new Parameters(new String[] { inputParam, "processname compress" });
		assertEquals(this.expectedFrom, parameters.getFromIdx());
		assertEquals(this.expectedTo, parameters.getToIdx());
	}

}
