#include "Word2Vec.h"

Word2Vec::Word2Vec(int method = 0, int n = 1000, float l = 0.5, float lf = 1)
{
    LFactor = lf;
    N = n;
    wordsSrc = sizeSrc = wordsTrg = sizeTrg = 0;
    L = l;
    Method = method;
    cout << "******* Method = " << method  << " ******** L: " << L << endl;
}

Word2Vec::~Word2Vec(){
    if(Method == 2){
        free(vocabIdxSrc);
        free(vocabIdxTrg);
        free(distSum);
    //    for(int i = 0; i < wordsSrc; i++)
      //      free(probability[i]);
        //free(probability);
    }
    free(vocabSrc);
    free(vocabTrg);

    free(MSrc);
    free(MTrg);
    free(vec);
}

void Word2Vec::readWord2Vec(std::string vectorsFileSrc, std::string vectorsFileTrg, vcbList *elist, vcbList *flist)
{

    std::cout << "reading source word2vec ...." << std::endl;
    readWord2Vec(vectorsFileSrc, 1);
    std::cout << "reading target word2vec ...." << std::endl;
    readWord2Vec(vectorsFileTrg, 0);
    std::cout << "reading word2vecs finished" << std::endl;
    cout << wordsSrc << " " << wordsTrg << endl;

    if(Method == 1){
        bestw = new char*[N];
        for (int a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
        bestd = (float *)malloc(N * sizeof(float));
        vec = (float*)malloc(max_size * sizeof(float));

        std::cout << "computing similarities for words ...." << std::endl;
        computeSimilarWords(elist, flist);
    }else if(Method == 2){
        vec = (float*)malloc(max_size * sizeof(float));
       /* probability = new float*[wordsSrc];
        for(int i = 0; i < wordsSrc; i++){
            probability[i] = (float *)malloc(wordsTrg * sizeof(float));
        }*/
        std::cout << "computing similarities for words ...." << std::endl;
        distSum = new double[wordsSrc];
        vocabIdxSrc = new int[elist->size()];
        vocabIdxTrg = new int[flist->size()];
        readWordIndexes(elist, flist);
       // computeSimilarities();
        computeDistanceSums(elist);
    }

}

void Word2Vec::readWord2VecWithTransmat(std::string vectorsFileSrc, std::string vectorsFileTrg, std::string tmFile, vcbList *elist, vcbList *flist)
{

    std::cout << "reading target word2vec ...." << std::endl;
    readWord2VecWithTransmat(vectorsFileTrg, tmFile, 0);
    std::cout << "reading source word2vec ...." << std::endl;
    readWord2VecWithTransmat(vectorsFileSrc, tmFile, 1);
    std::cout << "reading word2vecs finished" << std::endl;
    cout << wordsSrc << " " << wordsTrg << endl;

    if(Method == 1){
        bestw = new char*[N];
        for (int a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
        bestd = (float *)malloc(N * sizeof(float));
        vec = (float*)malloc(max_size * sizeof(float));

        std::cout << "computing similarities for words ...." << std::endl;
        computeSimilarWords(elist, flist);
    }else if(Method == 2){
        vec = (float*)malloc(max_size * sizeof(float));
       /* probability = new float*[wordsSrc];
        for(int i = 0; i < wordsSrc; i++){
            probability[i] = (float *)malloc(wordsTrg * sizeof(float));
        }*/
        std::cout << "computing similarities for words ...." << std::endl;
        distSum = new double[wordsSrc];
        vocabIdxSrc = new int[elist->size()];
        vocabIdxTrg = new int[flist->size()];
        readWordIndexes(elist, flist);
       // computeSimilarities();
        computeDistanceSums(elist);
    }

}

void Word2Vec::readWordIndexes(vcbList *elist, vcbList *flist){
    int b;
    for (b = 0; b < elist->size(); b++)
        vocabIdxSrc[b] = -1;
    for (b = 0; b < wordsSrc; b++)  {
        int idx = elist->getVocabId(&vocabSrc[b * max_w]);
        if(idx != -1)
            vocabIdxSrc[idx] = b;
    }

    for (b = 0; b < flist->size(); b++)
        vocabIdxTrg[b] = -1;
    for (b = 0; b < wordsTrg; b++)  {
        int idx = flist->getVocabId(&vocabTrg[b * max_w]);
        if(idx != -1)
            vocabIdxTrg[idx] = b;
    }

}

void Word2Vec::readWord2Vec(std::string vectorsFile, int isSrc){
    FILE *f;
    long long a, b;
    float len;
    char *vocab;
    float *M;
    long long words, size;
    std::map<std::string, long long> &dic = dicSrc;
    f = fopen(vectorsFile.c_str(), "rb");
    if (f == NULL) {
        printf("Word2Vec file not found\n");
        exit(EXIT_FAILURE);
    }
    if(isSrc){


        fscanf(f, "%lld", &wordsSrc);
        fscanf(f, "%lld", &sizeSrc);
        vocabSrc = (char *)malloc((long long)wordsSrc * max_w * sizeof(char));

        MSrc = (float *)malloc((long long)wordsSrc * (long long)sizeSrc * sizeof(float));

        words = wordsSrc;
        size = sizeSrc;
        vocab = vocabSrc;
        M = MSrc;
        dic = dicSrc;
    }else{

        fscanf(f, "%lld", &wordsTrg);
        fscanf(f, "%lld", &sizeTrg);
        vocabTrg = (char *)malloc((long long)wordsTrg * max_w * sizeof(char));

        MTrg = (float *)malloc((long long)wordsTrg * (long long)sizeTrg * sizeof(float));

        words = wordsTrg;
        size = sizeTrg;
        vocab = vocabTrg;
        M = MTrg;
        dic = dicTrg;
    }

    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
        exit(EXIT_FAILURE);
    }

    for (b = 0; b < words; b++) {
        a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
        }

        vocab[b * max_w + a] = 0;
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
    }
    fclose(f);
    char *tmp = (char *)malloc(max_size * sizeof(char));


    for (b = 0; b < words; b++) {
        //cout << b << endl;
        strcpy(tmp, &vocab[b * max_w]);
        dic[tmp] = b;
    }
    free(tmp);
}

void Word2Vec::readWord2VecWithTransmat(std::string vectorsFile, std::string tmFile, int isSrc){
    FILE *f;
    long long a, b;
    float len;
    char *vocab;
    float *M;
    long long words, size;
    std::map<std::string, long long> &dic = dicSrc;
    f = fopen(vectorsFile.c_str(), "rb");
    FILE *fmat = fopen(tmFile.c_str(), "rb");
    if (f == NULL) {
        printf("Word2Vec file not found\n");
        exit(EXIT_FAILURE);
    }
    if(isSrc){


        fscanf(f, "%lld", &wordsSrc);
        fscanf(f, "%lld", &sizeSrc);
        vocabSrc = (char *)malloc((long long)wordsSrc * max_w * sizeof(char));

        MSrc = (float *)malloc((long long)wordsSrc * (long long)sizeTrg * sizeof(float));
        tmat = (float *)malloc((long long)sizeSrc * (long long)sizeTrg * sizeof(float));
        words = wordsSrc;
        size = sizeSrc;
        vocab = vocabSrc;
        M = MSrc;
        dic = dicSrc;
        for(int a = 0; a < sizeSrc; a++){
            for(int b = 0; b < sizeTrg; b++){
                fread(&tmat[b + a * sizeTrg], sizeof(float), 1, fmat);
            }
        }
    }else{

        fscanf(f, "%lld", &wordsTrg);
        fscanf(f, "%lld", &sizeTrg);
        vocabTrg = (char *)malloc((long long)wordsTrg * max_w * sizeof(char));

        MTrg = (float *)malloc((long long)wordsTrg * (long long)sizeTrg * sizeof(float));

        words = wordsTrg;
        size = sizeTrg;
        vocab = vocabTrg;
        M = MTrg;
        dic = dicTrg;
    }

    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
        exit(EXIT_FAILURE);
    }

    for (b = 0; b < words; b++) {
        a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
        }
        float tmpM[size];
        vocab[b * max_w + a] = 0;
        for (a = 0; a < size; a++) fread(&tmpM[a], sizeof(float), 1, f);

        if(isSrc == 1){

            for (a = 0; a < sizeTrg; a++) {
                M[a + b * size] = 0;
                for (int c = 0; c < size; c++){
                    M[a + b * size] += tmpM[c] * tmat[a + c * sizeTrg];
                }
            }
        }else{
            for (a = 0; a < size; a++) M[b * size + a] = tmpM[a];
        }
        len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
    }
    fclose(f);
    char *tmp = (char *)malloc(max_size * sizeof(char));


    for (b = 0; b < words; b++) {
        //cout << b << endl;
        strcpy(tmp, &vocab[b * max_w]);
        dic[tmp] = b;
    }
    free(tmp);
}

void Word2Vec::getVector(std::string s, std::vector<std::string> &wordVec, std::vector<float> &distVec){
    wordVec.clear();
    distVec.clear();
    char **bestw;
    bestw = new char*[N];
    float *bestd;
    float*vec;
    float dist,len;
    long long a, b, c, d, bi;

    for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
    bestd = (float *)malloc(N * sizeof(float));
    vec = (float*)malloc(max_size * sizeof(float));

    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;

    for (b = 0; b < wordsSrc; b++)  if (!strcmp(&vocabSrc[b * max_w], s.c_str())) break;

    if (b == wordsSrc) b = -1;
    bi = b;
    if (bi == -1)
        return;

    for (a = 0; a < sizeSrc; a++) vec[a] = 0;
    for (a = 0; a < sizeSrc; a++) vec[a] += MSrc[a + bi * sizeSrc];

    len = 0;
    for (a = 0; a < sizeSrc; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    //for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = -1;
    for (a = 0; a < N; a++) bestw[a][0] = 0;

    for (c = 0; c < wordsTrg; c++) {
        a = 0;
       // if (bi == c) continue;
        dist = 0;
        for (a = 0; a < sizeTrg; a++) dist += vec[a] * MTrg[a + c * sizeTrg];
        for (a = 0; a < N; a++) {
            if (dist > bestd[a]) {
                for (d = N - 1; d > a; d--) {
                    bestd[d] = bestd[d - 1];
                    strcpy(bestw[d], bestw[d - 1]);
                }
                bestd[a] = dist;
                strcpy(bestw[a], &vocabTrg[c * max_w]);
                break;
            }
        }
    }

    for (a = 0; a < N; a++)
        if(bestd[a] != -1){
            wordVec.push_back(bestw[a]);
            distVec.push_back(bestd[a]);
        }
    for (a = 0; a < N; a++) free(bestw[a]);
    free(bestd);
    free(vec);
    free(bestw);

    return;
}

void Word2Vec::computeDistanceSums(vcbList* elist){
    double dist;
    long long a, b, c, bi;

    for (b = 0; b < wordsSrc; b++){
        WordIndex sIdx = elist->getVocabId(&vocabSrc[b * max_w]);
        if(sIdx == -1){
            distSum[b] = 0.0000001;
            continue;
        }
        bi = b;
        distSum[b] = 0;
        if(b % 1000 == 0)
            cout << b << endl;
        for (a = 0; a < sizeSrc; a++) vec[a] = 0;
        for (a = 0; a < sizeSrc; a++) vec[a] += MSrc[a + bi * sizeSrc];

        double len = 0;
        for (a = 0; a < sizeSrc; a++) len += vec[a] * vec[a];
        len = sqrt(len);
       // for (a = 0; a < sizeSrc; a++) vec[a] /= len;

        for (c = 0; c < wordsTrg; c++) {
            a = 0;
           // if (bi == c) continue;
            dist = 0;

            for (a = 0; a < sizeTrg; a++) dist += vec[a] * MTrg[a + c * sizeTrg];

            if(dist > 1.)
                cout << "+++++++++ " << dist << endl;
            distSum[b] += (1. + dist);
        }
    }
}

void Word2Vec::computeSimilarity(WordIndex s, WordIndex t){

    if(isComputed(s, t))
        return;
    int b = vocabIdxSrc[s], c = vocabIdxTrg[t], a;
    if(b == -1 || c == -1){
        insertW2V(s, t, 0.0);
        return;
    }
    for (a = 0; a < sizeSrc; a++) vec[a] = 0;
    for (a = 0; a < sizeSrc; a++) vec[a] += MSrc[a + b * sizeSrc];

    double len = 0;
    for (a = 0; a < sizeSrc; a++) len += vec[a] * vec[a];
    len = sqrt(len);
   // for (a = 0; a < sizeSrc; a++) vec[a] /= len;

    double dist = 0;
    for (a = 0; a < sizeTrg; a++) dist += vec[a] * MTrg[a + c * sizeTrg];
    if(dist > 1.)
        cout << "+++++++++ " << dist << endl;
    dist += 1.;
   // cout << dist << " ";
    dist /= distSum[b];
    //cout << dist << endl;
    insertW2V(s, t, dist);

}

void Word2Vec::computeSimilarities(sentenceHandler& sHandler1){
    WordIndex i, j;

    cout << "Initialize Word2Vec Probabilities\n";

    sentPair sent ;
    sHandler1.rewind();
    int sNum = 0;
    while(sHandler1.getNextSentence(sent)){
      Vector<WordIndex>& es = sent.eSent;
      Vector<WordIndex>& fs = sent.fSent;
      for( i=0; i < es.size(); i++){
        for(j=1; j < fs.size(); j++){
            computeSimilarity(es[i], fs[j]);
           // cout << getW2VProb(es[i], fs[j]);
        }
      }
      if(sNum % 1000 == 0)
          cout << "Sentence " << sNum << endl;
      sNum++;
    }
}

/*
void Word2Vec::computeSimilarities(){
    float dist,sumd;
    long long a, b, c, bi;

    for (b = 0; b < wordsSrc; b++){
        bi = b;
        sumd = 0;
        if(b % 1000 == 0)
            cout << b << endl;
        for (a = 0; a < sizeSrc; a++) vec[a] = 0;
        for (a = 0; a < sizeSrc; a++) vec[a] += MSrc[a + bi * sizeSrc];
        for (c = 0; c < wordsTrg; c++) {
            a = 0;
           // if (bi == c) continue;
            dist = 0;

            for (a = 0; a < sizeTrg; a++) dist += vec[a] * MTrg[a + c * sizeTrg];
            sumd += dist;
            probability[b][c] = dist;
        }
        for (c = 0; c < wordsTrg; c++)
            probability[b][c] /= sumd;
    }

}
*/

void Word2Vec::computeSimilarWords(vcbList* elist, vcbList* flist){
    similarWords.resize(elist->size());
    for(int i = 0; i < wordsSrc; i++){
       WordIndex sIdx = elist->getVocabId(&vocabSrc[i * max_w]);
       if(sIdx == -1)
           continue;
       std::vector<std::string> sim = getVector(&vocabSrc[i * max_w]);

       for(int j = 0; j < sim.size(); j++){
            WordIndex tIdx = flist->getVocabId(sim[j]);
            if(tIdx == -1)
                continue;
            similarWords[sIdx][tIdx] = true;
       }
     if( i % 1000 == 0)  cout << i << endl;

    }
}

std::vector<std::string> Word2Vec::getVector(std::string s){
    std::vector<std::string> wordVec;
    //std::vector<float> distVec;

    float dist,len;
    long long a, b, c, d, bi;


    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;

    for (b = 0; b < wordsSrc; b++)  if (!strcmp(&vocabSrc[b * max_w], s.c_str())) break;

    if (b == wordsSrc) b = -1;
    bi = b;
    if (bi == -1)
        return wordVec;

    for (a = 0; a < sizeSrc; a++) vec[a] = 0;
    for (a = 0; a < sizeSrc; a++) vec[a] += MSrc[a + bi * sizeSrc];

    len = 0;
    for (a = 0; a < sizeSrc; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    //for (a = 0; a < size; a++) vec[a] /= len;
    for (a = 0; a < N; a++) bestd[a] = -1;
    for (a = 0; a < N; a++) bestw[a][0] = 0;

    for (c = 0; c < wordsTrg; c++) {
        a = 0;
       // if (bi == c) continue;
        dist = 0;
        for (a = 0; a < sizeTrg; a++) dist += vec[a] * MTrg[a + c * sizeTrg];
        for (a = 0; a < N; a++) {
            if (dist > bestd[a]) {
                for (d = N - 1; d > a; d--) {
                    bestd[d] = bestd[d - 1];
                    strcpy(bestw[d], bestw[d - 1]);
                }
                bestd[a] = dist;
                strcpy(bestw[a], &vocabTrg[c * max_w]);
                break;
            }
        }
    }

    for (a = 0; a < N; a++)
        if(bestd[a] != -1){
            wordVec.push_back(bestw[a]);
          //  distVec.push_back(bestd[a]);
        }

    return wordVec;
}

std::map<WordIndex, bool> Word2Vec::getVectorMap(WordIndex s){

    return similarWords[s];
}

/*
float Word2Vec::getProbability(WordIndex s, WordIndex t){
    if(vocabIdxSrc[s] == -1 || vocabIdxTrg[t] == -1)
        return 0.0;
    return probability[vocabIdxSrc[s]][vocabIdxTrg[t]];
}
*/
