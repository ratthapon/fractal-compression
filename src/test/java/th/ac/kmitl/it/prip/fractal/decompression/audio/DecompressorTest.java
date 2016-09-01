package th.ac.kmitl.it.prip.fractal.decompression.audio;

import static org.junit.Assert.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import th.ac.kmitl.it.prip.fractal.Parameters;
import th.ac.kmitl.it.prip.fractal.dataset.DataSetManager;

@RunWith(Parameterized.class)
public class DecompressorTest {
	private int input;
	private double[] expected;

	@Parameterized.Parameters
	public static Collection<Object[]> codeSamplesMap() {
		return Arrays.asList(new Object[][] { { 0,
				new double[] { 119.958911660032, 413.118470465428, 361.806151350765, 335.118535867823, 43.6287225762808,
						97.3570353754950, 87.9528585406001, 83.0617318163834, 31.1149103412526, 53.2105274322088,
						49.3430862345668, 47.3316241652941, 24.2255822356363, 35.8454231985969, 33.2330420157458,
						29.5960007171595, 24.6662136367885, 25.6079453482101, 28.0343323547106, 18.7597573196174,
						20.2952164364747, 21.0636309829143, 22.5410628068425, 14.1001412281475, 16.9671693749009,
						17.9017396965181, 19.6986394835584, 9.43252097877791, 14.4946080638870, 15.4556921624863,
						17.3035700290899, 6.74620155604889, 18.2771660823973, 7.68326872530884, 9.53754031744122,
						10.5019498024460, 15.7520929977698, 6.76056426417598, 8.33436998822230, 9.15290874719644,
						13.0240731339764, 6.34294762635294, 7.51235873301879, 8.12057123431775, 9.63687827216704,
						5.90576892731976, 6.55883265184799, 6.89849209166245, 6.52659240298662, 6.01689005957241,
						5.16283287421646, 8.03006016847933, 4.65988230377235, 4.33122872278797, 3.69932597986864,
						7.30953757268698, 3.22309930627382, 2.71395866793912, 1.73503318375071, 7.32786874448112,
						2.16478069428405, 1.59928700476783, 0.512011422866138, 6.72387702141589, 0.888844380731348,
						0.298288548010610, -0.837174054969443, 5.64999552677995, -3.50744702093342, 2.75689425822521,
						1.66043375027118, 1.09016296669737, -3.90849235226299, 1.02878428660062, 0.164602625654295,
						-0.284859575093426, -3.65036229814636, -0.679763236383634, -1.19971328556512, -1.47014009967742,
						-2.05656913439724, -1.79441625683770, -2.67451175711149, -3.52083652928944, -2.96369411479306,
						-2.68703853590827, -2.15511243117022, -5.19413356435301, -3.71665355980804, -3.20128415670086,
						-2.21038261695588, -7.87164024910634, -4.34163059271053, -3.70545256984001, -2.48227211206804,
						-9.47059479026377, -5.00393410781938, -4.28301554109462, -2.89690431002150, -10.8160894875334,
						-2.18391003562943, -10.1030952131413, -8.71698398206823, -7.99606541534346, -3.57991835067023,
						-10.8800738961131, -9.60231269173393, -8.93774713925344, -6.52436683573553, -11.5665944300667,
						-10.6840429880350, -10.2250266361093, -7.59751651967136, -12.7140868566112, -13.5020336595294,
						-11.5996512973065, -13.9396399661615, -14.3644621056176, -15.1812682830252, -10.5146591191859,
						-17.3963965987550, -18.2625783476840, -19.9279873213989, -10.4131073006031, -22.3840296755546,
						-23.7214103450932, -26.2927942042847, -11.6018628512233, -28.8781122160361, -30.6929205340195,
						-34.1822552632248, -14.2468509122281, -39.5361484476659, -41.7452426620065, -45.9926714873002,
						-21.7260941436034, -60.1310773029685, -62.6734741678556, -67.5617445092276, -39.6338862224909,
						-108.815050470444, -100.484700789769, -148.077982076161, -140.620949626037, -179.836349552453,
						-261.750762043636, 261.054482693865, 169.546807958286, 162.335826924933, 131.310976492268,
						101.476601170371, 108.183987350740, 65.4945740782034, 68.8456810135136, 47.4577929950902,
						51.2013589209706, 24.3629763545682, 44.8984431361669, 41.3040782484594, 39.4346435847450,
						14.9664231404821, 33.7770082547881, 30.4845529446083, 28.7721419129319, 10.9523963993308,
						26.6585603369406, 23.9094782353100, 22.4796764386808, 9.17834580050092, 20.6233741085999,
						18.6201273196683, 17.5782356074465, 8.86052214243387, 16.1128381096929, 14.8434503648458,
						14.1832398614472, 8.17297784784241, 12.5387033071839, 11.5571977017430, 10.1907139568683,
						9.50886733715827, 9.75900910212365, 10.2399570562791, 7.49218629710714, 8.75246637625086,
						9.23799654209594, 10.1715261347102, 4.83804816636828, 7.65836893045231, 8.29454695332283,
						9.51772741109481, 2.52940473289907, 6.46518535140236, 7.15626468086502, 8.48500396480745,
						0.893598451104058, 9.39671305720925, 2.13308639581388, 3.40445387466421, 4.06569404073925,
						8.21713514668853, 1.62501460947163, 2.77884699353705, 3.37895697743573, 6.45226341815709,
						1.44656511362368, 2.32272275910727, 2.77841368769023, 3.77475673340376, 1.39129492783802,
						1.80847713831468, 2.02545424267525, 2.13987522297451, 1.52052302678069, 0.482736561287697,
						3.96677689312795, 0.408380069276390, 0.0782748175771111, -0.556419054846453, 3.06973892878306,
						-0.776900455307603, -1.28604109364230, -2.26496657783071, 3.32786898289969, -2.11459691847063,
						-2.70847813759682, -3.85033446610567, 2.67336399969117, -3.11115550005936, -3.70171133278010,
						-4.83717393576015, 1.64999564598924, -7.72387678299731, -1.51201118444756, -2.59928676634925,
						-3.16478045586547, -7.90849191353186, -2.97121568041851, -3.83539727034554, -4.28485943415601,
						-7.89016094205028, -4.93550827109873, -5.45266719077598, -5.72164233417338, -7.43176990159325,
						-6.12923113742812, -6.66872901928487, -10.1719843772952, -8.30585142300200, -7.87667427140121,
						-7.05149470548091, -11.7659430417538, -9.27642880799850, -8.57724692419831, -7.23292884663615,
						-14.9133397364964, -11.0293884914321, -10.0914927834095, -8.28819927084039, -18.5908466596683,
						-12.9093096106227, -11.8553824138316, -9.82899498030715, -21.4062313288212, -8.90744924844979,
						-20.7759355652244, -18.6985700521367, -17.6181290783634, -12.5198850307757, -23.8218173960901,
						-21.8436169836440, -20.8147519300574, -18.8233148235883, -27.6969666158091, -26.1437931346841,
						-25.3359852367417, -27.6767442620790, -31.4682012283045, -35.4109479434569, -34.2365272169061,
						-42.3521474853075, -48.9336968793762, -53.6610327492639, -32.6338980618811, -67.9623160593794,
						-88.2879979345619, -102.887345178703, -37.9496193010362, -313.676232048942, -348.874697324533,
						-416.550840355070, -29.9008394430355, 845.555764484016, 111.263130512371, -1.81721339830040,
						271.198745259132, 117.430572138200, 56.5752658302977, 67.2268949629657, 72.7668235310806,
						64.8115523062298, 36.1873908741692, 41.1975365838185, 43.8033210137493, 41.5836504485908,
						28.1659025363288, 30.5144384854197, 31.7359156168108, 32.2561966659831, 22.4147753813412,
						20.8992061764347, 24.5583264539260, 19.2908674184656, 18.8472119375147, 17.9941948656046,
						22.8676855354732, 15.2275601730479, 14.4811573442684, 13.0460474835580, 21.2451735581952,
						12.1074822459128, 11.2450479283649, 9.58684414669910, 19.0605591810224, 4.72301001383244,
						14.1442493675469, 12.4952305119334, 11.6375732884431, 3.11901825266118, 11.9212271628060,
						10.3805584759251, 9.57925434055356, 2.83337711678322, 9.73733133274772, 8.52891806003655,
						7.90042047265341, 2.77782703236000, 7.42385329721279, 6.61064978684905, 6.18770140002075,
						3.40482111506673, 5.11677740507937, 4.81713018285867, 4.66128320526859, 3.97454551890617,
						4.19152262326674, 4.60870483374340, 2.22524302817767, 3.22158631230977, 3.67727724089273,
						4.55343488637632, -0.452263418157090, 2.81078510744992, 3.35266832447652, 4.39454801480409,
						-1.55795999135720, 1.87254535454140, 2.47410704609222, 3.63073063068117, -2.97733670309716,
						4.97733658388787, -1.63073074989046, -0.474107165301508, 0.127454526249310, 3.58132854537168,
						-2.40770875058731, -1.35943526371164, -0.814226623342346, 2.03288689182851, -2.31725249083811,
						-1.55583866859660, -1.15982617760061, -0.134265475548051, -2.24078305250425, -1.91072869391541,
						-1.70769590044500, -1.95891781224923, -2.87253747236552, -3.52876395219633, -0.609876241102216,
						-4.18770163843933, -4.61065002526763, -5.42385353563137, -0.777827270778575, -5.36954054964973,
						-5.96819886283351, -7.11924011739469, -0.543065970988975, -6.95873922558059, -7.64358982720801,
						-8.96035312661316, -1.43736927857958, -8.23467577755187, -8.94458852238381, -10.3095388428673,
						-2.51125101163410, -13.8024286881746, -6.29539163670248, -7.60936373558411,
						-8.29276262955938 } } });
	}

	public DecompressorTest(int input, double[] expectedResult) {
		this.input = input;
		this.expected = expectedResult;
	}

	@Test
	public void testDecompress() throws IOException {
		String fileDir = "test-classes//expected//synth_code//";
		List<String> parameterList = Files.readAllLines(Paths.get("test-classes//input-param.txt"));
		String[] params = new String[parameterList.size()];
		parameterList.toArray(params);
		Parameters testParameters = new Parameters(params);
		testParameters.setParameter("processname", "decompress");
		testParameters.setParameter("fs", "8000");
		testParameters.setParameter("inpathprefix", fileDir);
		testParameters.setParameter("inext", "mat");
		testParameters.setParameter("outext", "raw");
		DataSetManager dataSetManager = new DataSetManager(testParameters);

		double[][] codes = dataSetManager.readCode(input);
		Decompressor decompressor = new Decompressor(codes, testParameters);
		double[] expected = this.expected;
		double[] actual = decompressor.decompress();
		assertArrayEquals(expected, actual, 1e0);
	}

}