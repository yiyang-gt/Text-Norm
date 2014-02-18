import java.io.*;
import java.util.*;

public class SmcSgdTraining {

	int targetVocabSize = 10000;
	int sourceVocabSize = 10000;

	int samplerNum = 10;
	int lSamplerNum = 1;
	
	int strSimNum = 1000; // the maximum string similar source words for each target word

	String basePath;
	Map<String, Double> featureWeights;
	ArrayList<String> trainSet;
	Set<String> nonstandardWords, standardWords;
	Map<String, Integer> bigramCount, trigramCount;
	Map <String, Map<String, Short>> stringSimWords; // store string similar source words for each target word

	public SmcSgdTraining(String basePath) {
		this.basePath = basePath;
	}

	public void initialize() throws Exception {

		featureWeights = new HashMap<String, Double>();
		stringSimWords = new HashMap <String, Map<String, Short>>();

		String s = "";
		BufferedReader br = null;
	
		trainSet = new ArrayList<String>();
		br = new BufferedReader(new FileReader(basePath + "/training-data/trainSents.txt"));
		while ((s = br.readLine()) != null) {
			if (s.trim().equals(""))
				continue;
			String sent = convert2PlusChar(s);
			trainSet.add(sent);
		}
		br.close();


		// read nonstandard words
		nonstandardWords = new HashSet <String>();
		int pos = 0;
		br = new BufferedReader(new FileReader(basePath + "/training-data/OOVDictionary.txt"));
		while ((s = br.readLine()) != null) {
			String[] strs = s.split("\\s+");
			nonstandardWords.add(strs[0]);
			if(pos++==sourceVocabSize) break;

		}
		br.close();
	
    	// read standard words
		pos=0;
		standardWords = new HashSet <String>();
		br = new BufferedReader(new FileReader(basePath + "/training-data/IVWordList.txt"));
		while ((s = br.readLine()) != null) {
			String[] strs = s.split("\\s+");
			standardWords.add(strs[0]);
			if(pos++==targetVocabSize) break;
		}
		br.close();


		// read LM counts
		bigramCount = new HashMap<String, Integer>();
		br = new BufferedReader(new FileReader(basePath
				+ "/language-models/edinburgh-bigram-punts-counts.txt"));
		while ((s = br.readLine()) != null) {
			if (s.contains("<unk>"))
				continue;
			String[] strs = s.split("\\s+");
			int countNum = Integer.parseInt(strs[2]);
			if (countNum > 0)
				bigramCount.put(strs[0] + " " + strs[1], countNum);
		}
		br.close();

		trigramCount = new HashMap<String, Integer>();
		br = new BufferedReader(new FileReader(basePath
				+ "/language-models/edinburgh-trigram-punts-counts.txt"));
		while ((s = br.readLine()) != null) {
			if (s.contains("<unk>"))
				continue;
			String[] strs = s.split("\\s+");
			int countNum = Integer.parseInt(strs[3]);
			if (countNum > 1)
				trigramCount.put(strs[0] + " " + strs[1] + " " + strs[2],
						countNum);
		}
		br.close();

		for(String stdStr : standardWords) {
            stringSimWords.put(stdStr, getSimNonWords(stdStr));
        }

	}

	public void update() throws Exception {
		Random rd = new Random();
        int iterNum=0;
		while(iterNum<1000000) {
			int randomPos = rd.nextInt(trainSet.size());
			Map <String, Double> gradients = getGradients(randomPos);
			for (String key : gradients.keySet()) {
				if (!featureWeights.containsKey(key)) {
					featureWeights.put(key, 0.0);
				}
				double tempWeight = featureWeights.get(key)
					+ gradients.get(key);
			    featureWeights.put(key, tempWeight);
				
			}
			System.out.println("IterNum = " + iterNum++);

		}

	}

	private int nonZeroFeatureNum() {
		int res=0;
		for(String key : featureWeights.keySet()) if(featureWeights.get(key)>0 || featureWeights.get(key)<0) res++;
		return res;
	}

	public Map<String, Double> getGradients(int tPos) {
        Map <String, Double> resGradients = new HashMap <String, Double>();	
        String sent = trainSet.get(tPos);

        String[] unigram = new String[samplerNum];
        String[] bigram = new String[samplerNum];
        double[] weights = new double[samplerNum];
        boolean unIllFlag = false, biIllFlag = false;


        // standard words candidates
        ArrayList<String> alStds = new ArrayList<String>(standardWords);
        // initalize particles
        for (int k = 0; k < samplerNum; k++) {
            unigram[k] = bigram[k] = "";
            weights[k] = 1.0 / samplerNum;
        }

        // split tokens
        ArrayList<Map<String, Double>> diffFeaturesMaps = new ArrayList<Map<String, Double>>();
        String word;
        String[] tokens = sent.split("\\s+");

        for (int wPos = 0; wPos < tokens.length; wPos++) {
            double this_weight_mass = 0;

            if (tokens[wPos].contains(":")) {
                word = tokens[wPos].substring(0, tokens[wPos].indexOf(":"));
                unIllFlag = biIllFlag;
                biIllFlag = true;

                for (int k = 0; k < samplerNum; k++) {
                    ArrayList<Double> mass4Sample = new ArrayList<Double>();
                    double tmpWeight = 0;
                    for (int i = 0; i < alStds.size(); i++) {
                        double lmScore = languageModel(alStds.get(i),
                                unigram[k] + " " + bigram[k]);
                        double new_mass=0.0;
                        if(lmScore>0) new_mass = Math.exp(featureExtr(alStds.get(i), word)) * lmScore;
                        mass4Sample.add(new_mass);
                        tmpWeight += new_mass;
                    }

                    double sum = 0;
                    for (double doub : mass4Sample) {
                        sum += doub;
                    }
                    if(sum==0) continue;
                    ArrayList<Double> probMass = new ArrayList<Double>();
                    double aggProb = 0;
                    for (int i = 0; i < mass4Sample.size(); i++) {
                        aggProb += mass4Sample.get(i) / sum;
                        probMass.add(aggProb);
                    }

                    int sampledPos = samplePos(probMass);
                    String sampledStd = alStds.get(sampledPos);


                    ArrayList<String> alOov = new ArrayList<String>(nonstandardWords);

                    // second factor of importance distribution
                    double total_prob_mass = 0;
                    // calu z(t_n) normalize
                    ArrayList<Double> mass4SampleInn = new ArrayList<Double>();
                    for (int j = 0; j < alOov.size(); j++) {
                        double local_prob_mass = Math.exp(featureExtr(
                                    sampledStd, alOov.get(j)));
                        total_prob_mass += local_prob_mass;
                        mass4SampleInn.add(local_prob_mass);
                    }
                    tmpWeight /= total_prob_mass;

                    double sumInn = 0;
                    for (double doub : mass4SampleInn) {
                        sumInn += doub;
                    }
                    if(sumInn==0) continue;
                    ArrayList<Double> probMassInn = new ArrayList<Double>();
                    double aggProbInn = 0;
                    for (int i = 0; i < mass4SampleInn.size(); i++) {
                        aggProbInn += mass4SampleInn.get(i) / sumInn;
                        probMassInn.add(aggProbInn);
                    }

                    // identify related features
                    Map<String, Double> mapGTFeatures = new HashMap<String, Double>();
                    mapGTFeatures.put(sampledStd + "|" + word, 1.0);
                    // string similarity feature
                    ArrayList<Integer> alGtStrSimFea = getStrSimFeature(sampledStd, word);
                    if(alGtStrSimFea!=null) {
                        for(int gtStrSimFea : alGtStrSimFea) mapGTFeatures.put("string-similarity-" + gtStrSimFea, 1.0);
                    }				

                    Map<String, Double> mapOtherMaxFeatures = new HashMap<String, Double>();
                    for(int l=0; l<lSamplerNum; l++) {
                        // sample negative training sample
                        int sampledPos_ill = samplePos(probMassInn);
                        String sampledIll = alOov.get(sampledPos_ill);

                        String feaStr = sampledStd + "|" + sampledIll;
                        if(!mapOtherMaxFeatures.containsKey(feaStr)) mapOtherMaxFeatures.put(feaStr, 0.0);
                        mapOtherMaxFeatures.put(feaStr, mapOtherMaxFeatures.get(feaStr)+1.0/lSamplerNum);
                        ArrayList <Integer> alOtherStrSimFea = getStrSimFeature(sampledStd, sampledIll);
                        if(alOtherStrSimFea!=null) {
                            for(int otherStrSimFea : alOtherStrSimFea) {
                                feaStr = "string-similarity-" + otherStrSimFea;
                                if(!mapOtherMaxFeatures.containsKey(feaStr)) mapOtherMaxFeatures.put(feaStr, 0.0);
                                mapOtherMaxFeatures.put(feaStr, mapOtherMaxFeatures.get(feaStr)+1.0/lSamplerNum);
                            }
                        }

                    }

                    Map<String, Double> diffFeatures = featuresMinus(
                            mapGTFeatures, mapOtherMaxFeatures);
                    diffFeaturesMaps.add(diffFeatures);
                    double thisWeight = weights[k] * tmpWeight;
                    //						System.out.println(sampledStd+" "+word+" "
                    //								/*+sampledIll*/+"\t"+thisWeight+"\t"+unigram[k]
                    //								+" "+bigram[k]+"\t"
                    //								+languageModel(sampledStd, unigram[k]+" "+bigram[k]));
                    this_weight_mass += thisWeight;

                    unigram[k] = bigram[k];
                    bigram[k] = sampledStd;

                    weights[k] = thisWeight;

                }

            } else {
                word = tokens[wPos];
                for (int k = 0; k < samplerNum; k++) {
                    if (!unigram[k].equals("") && (unIllFlag || biIllFlag)) {
                        weights[k] = weights[k]
                            * languageModel(word, unigram[k] + " "
                                    + bigram[k]);
                    }
                    this_weight_mass += weights[k];
                    unigram[k] = bigram[k];
                    bigram[k] = word;
                }
                unIllFlag = biIllFlag;
                biIllFlag = false;

            }
            if (this_weight_mass == 0) {
                continue;
            }
            for (int k = 0; k < samplerNum; k++)
					weights[k] = weights[k] / this_weight_mass;
			}

			//add gradient 
			for (int k = 0; k < samplerNum; k++) {
				for (int m = 0; m < diffFeaturesMaps.size() / samplerNum; m++) {
					Map<String, Double> diffFeatures = diffFeaturesMaps.get(m
							* samplerNum + k);
					for (String key : diffFeatures.keySet()) {
						if(!resGradients.containsKey(key)) resGradients.put(key, 0.0);
						resGradients.put(key, resGradients.get(key) + weights[k]*diffFeatures.get(key));
					}
				}
			}

		return resGradients;
	}

	private Map <String, Short> getSimNonWords(String stdStr) {
                ArrayList <Item> alStrSimNonWords = new ArrayList <Item>();
                for(String nonStr : nonstandardWords) {
                        alStrSimNonWords.add(new Item(nonStr, ((double)LCS(stdStr, nonStr)/stdStr.length()) / computeLevenshteinDistance(stdStr, nonStr)));
                }
                Collections.sort(alStrSimNonWords);
                Map <String, Short> map = new HashMap<String, Short>();
                for(short i=0; i<strSimNum; i++) map.put(alStrSimNonWords.get(i).getKey(), (short)(i+1));
                return map;
        }

	private ArrayList<Integer> getStrSimFeature(String word1, String word2) {
		if(!stringSimWords.get(word1).containsKey(word2)) return null;
		ArrayList <Integer> alRes = new ArrayList <Integer>();		
		short rank = stringSimWords.get(word1).get(word2);
		if(rank<=5) alRes.add(5);
		else if(rank<=10) alRes.add(10);
		else if(rank<=25) alRes.add(25);
		else if(rank<=50) alRes.add(50);
		else if(rank<=100) alRes.add(100);
		else if(rank<=250) alRes.add(250);
		else if(rank<=500) alRes.add(500);
		else if(rank<=1000) alRes.add(1000);
		return alRes;
	}

	private double featureExtr(String word1, String word2) {

		double local_prob_mass = 0.0;
		// word pair features
		String featureStr = word1 + "|" + word2;
		if (featureWeights.containsKey(featureStr)) {
			local_prob_mass += featureWeights.get(featureStr);
		}
    
        // string similarity features
		ArrayList <Integer> alStrSimFea = getStrSimFeature(word1, word2);
		if(alStrSimFea!=null) {
			for(int strSimFea : alStrSimFea) {
				featureStr = "string-similarity-" + strSimFea;
				if (featureWeights.containsKey(featureStr))
					local_prob_mass += featureWeights.get(featureStr);
			}
		}


		return local_prob_mass;
	}

	private Map<String, Double> featuresMinus(Map<String, Double> map1,
			Map<String, Double> map2) {
		Map<String, Double> resMap = new HashMap<String, Double>();
		for (String key : map1.keySet()) {
			if (map2.containsKey(key) && map1.get(key) != map2.get(key))
				resMap.put(key, map1.get(key) - map2.get(key));
			if (!map2.containsKey(key))
				resMap.put(key, map1.get(key));
		}
		for (String key : map2.keySet()) {
			if (!map1.containsKey(key))
				resMap.put(key, -1.0 * map2.get(key));
		}
		return resMap;

	}

	private int samplePos(ArrayList<Double> probMass) {
		
		double randomDoub = new Random().nextDouble();
		int low = 0, high = probMass.size()-1, mid = (low+high)/2;
		while(low<high) {
			if(mid==0) return mid;
			if(randomDoub>=probMass.get(mid-1) && randomDoub<probMass.get(mid)) return mid;
			if(randomDoub<probMass.get(mid-1)) {
				high=mid-1;
			} else if(randomDoub>=probMass.get(mid)) {
				low=mid+1;
			}
			mid=(low+high)/2;
		}
		
		return mid;
	}

    // this is a simple unsmoothed LM, can be replaced by any transition models
	private double languageModel(String strN, String strP) {
		double result = 0;
		if (trigramCount.containsKey(strP + " " + strN))
			result = (double) trigramCount.get(strP + " " + strN)
					/ bigramCount.get(strP);
		return result;
	}

	private int minimum(int a, int b, int c) {
		return Math.min(Math.min(a, b), c);
	}

	public void outputFeatureWeights(String path) throws Exception {
		BufferedWriter bw = new BufferedWriter(new FileWriter(path));
		for(String featureStr :  featureWeights.keySet()) {
			bw.append(featureStr+"\t"+featureWeights.get(featureStr));
			bw.newLine();
		}
		bw.close();
	}


	public int computeLevenshteinDistance(String str1, String str2) {
		int[][] distance = new int[str1.length() + 1][str2.length() + 1];

		for (int i = 0; i <= str1.length(); i++)
			distance[i][0] = i;
		for (int j = 1; j <= str2.length(); j++)
			distance[0][j] = j;

		for (int i = 1; i <= str1.length(); i++)
			for (int j = 1; j <= str2.length(); j++)
				distance[i][j] = minimum(
						distance[i - 1][j] + 1,
						distance[i][j - 1] + 1,
						distance[i - 1][j - 1]
								+ ((str1.charAt(i - 1) == str2.charAt(j - 1)) ? 0
										: 1));
		return distance[str1.length()][str2.length()];
	}
	

	private int LCS(String x, String y) {

		int i, j;
		int lenx = x.length();
		int leny = y.length();
		int[][] table = new int[lenx + 1][leny + 1];

		// Initialize table that will store LCS's of all prefix strings.
		// This initialization is for all empty string cases.
		for (i = 0; i <= lenx; i++)
			table[i][0] = 0;
		for (i = 0; i <= leny; i++)
			table[0][i] = 0;

		// Fill in each LCS value in order from top row to bottom row,
		// moving left to right.
		for (i = 1; i <= lenx; i++) {
			for (j = 1; j <= leny; j++) {

				// If last characters of prefixes match, add one to former
				// value.
				if (x.charAt(i - 1) == y.charAt(j - 1))
					table[i][j] = 1 + table[i - 1][j - 1];

				// Otherwise, take the maximum of the two adjacent cases.
				else
					table[i][j] = Math.max(table[i][j - 1], table[i - 1][j]);
			}
		}
		return table[lenx][leny];
	}
	
	
	private String convert2PlusChar(String str) {
		if(str.equals("")) return str;
		String tmp =str.charAt(0)+"";
		int repetCount=1;
		
		for(int i=1; i<str.length(); i++) {
			if(str.charAt(i) == str.charAt(i-1)) repetCount++;
			else repetCount=1;
			if(repetCount>2) {
				continue;
			}
			tmp += str.charAt(i)+"";
		}
		return tmp;
	}
	
	public static void main(String [] args) throws Exception {
		if(args.length < 2) {
			System.out.println("input regularization parameter lambda and output path"); 
			System.exit(-1);
		}
		SmcSgdTraining st = new SmcSgdTraining("/nethome/yyang319/data/text-norm");
		st.initialize();
		st.update();
	}
}
