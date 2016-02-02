#ifndef WORD2VEC_H
#define WORD2VEC_H

#include<vector>
#include<string>
#include <map>
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <iostream>
#include <stdlib.h>
#include <boost/algorithm/string.hpp>
#include "vocab.h"
#include "defs.h"
#include "getSentence.h"

class Word2Vec
{

public:
    static const long long max_size = 2000;         // max length of strings
    long long N;                  // number of closest words that will be shown
    float L;

    float LFactor;
    int Method;
    static const long long max_w = 200;              // max length of vocabulary entries
    long long wordsSrc, sizeSrc, wordsTrg, sizeTrg;
    float *MSrc, *MTrg;
    char *vocabSrc, *vocabTrg;
    //std::map<std::string, std::map<std::string, float> > cosineDistance;
    std::map<std::string, long long> dicSrc, dicTrg;

    std::vector<std::map<WordIndex, bool > > similarWords;
    char **bestw;
    float *bestd;
    float*vec;
    float *tmat;
    //float **probability;
    int *vocabIdxSrc, *vocabIdxTrg;
    double *distSum;
    hash_map<wordPairIds, PROB, hashpair, equal_to<wordPairIds> > wv;

public:
    Word2Vec();
    Word2Vec(int method, int n, float l, float lf);
    ~Word2Vec();
    void readWord2Vec(std::string vectorsFileSrc, std::string vectorsFileTrg, vcbList* elist=0, vcbList* flist=0);
    void readWord2Vec(std::string vectorsFile, int isSrc);
    void readWord2VecWithTransmat(std::string vectorsFileSrc, std::string vectorsFileTrg, std::string tmFile, vcbList *elist, vcbList *flist);
    void readWord2VecWithTransmat(std::string vectorsFile, std::string tmFile, int isSrc);

    void computeSimilarWords(vcbList* elist=0, vcbList* flist=0);
  //  void computeSimilarities();
    void computeDistanceSums(vcbList* elist);
    void computeSimilarities(sentenceHandler& sHandler1);
    void computeSimilarity(WordIndex s, WordIndex t);
    void readWordIndexes(vcbList *elist, vcbList *flist);
    void getVector(std::string s, std::vector<std::string> &wordVec, std::vector<float> &distVec);
    std::vector<std::string> getVector(std::string s);
    std::map<WordIndex, bool> getVectorMap(WordIndex s);
  //  float getProbability(WordIndex s, WordIndex t);

    void insertW2V(WordIndex e, WordIndex f, PROB pval = 0.0){
        wv[wordPairIds(e, f)] = pval ;
    }

    bool isComputed(WordIndex e, WordIndex f){
        typename hash_map<wordPairIds, PROB, hashpair, equal_to<wordPairIds> >::const_iterator i= wv.find(wordPairIds(e, f));
        if(i == wv.end())
            return false;
        return true;
    }

    PROB getW2VProb(WordIndex e, WordIndex f) const
      {
        typename hash_map<wordPairIds, PROB, hashpair, equal_to<wordPairIds> >::const_iterator i= wv.find(wordPairIds(e, f));
        if(i == wv.end())
          return 0.0;
        else
          return (*i).second;
      }


};

#endif // WORD2VEC_H
