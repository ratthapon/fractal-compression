package th.ac.kmitl.it.prip.fractal;

import static org.junit.Assert.*;
import static th.ac.kmitl.it.prip.fractal.Parameters.ProcessName.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import th.ac.kmitl.it.prip.fractal.Parameters.ProcessName;

@RunWith(Parameterized.class)
public class ParametersSettingTest {
	private String input;
	private ProcessName expected;

	@Parameterized.Parameters
	public static Collection<Object[]> setDataset() {
		return Arrays.asList(new Object[][] { { "compress", COMPRESS }, { "decompress", DECOMPRESS },
				{ "distributed_compress", DISTRIBUTED_COMPRESS },
				{ "distributed_decompress", DISTRIBUTED_DECOMPRESS } });
	}

	public ParametersSettingTest(String input, ProcessName expected) {
		this.input = input;
		this.expected = expected;
	}

	@Test
	public void testSetProcessName() throws IOException {
		String inputParam = String.format("processname %s", input);
		Parameters parameters = new Parameters(new String[] { inputParam });
		assertEquals(expected, parameters.getProcessName());
	}

}
