#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <filesystem>
#include "Diagnostico.h"
#include <err.h>
#include <random>
#include <omp.h>
#include <atomic>
#include <iomanip>

//Compilar con: g++ file.cpp utils/Diagnostico.h -o programa -std=c++17 -fopenmp -pthread
//Correr con: OMP_NUM_THREADS=8 ./programa
using json = nlohmann::json;
using namespace std;

class Lda{
    private:
        //Matrices. Documents x Topics and Topics x Words
        vector<vector<int>> documentsPerTopicMatrix;
        vector<vector<int>> topicPerWordMatrix;

        //Corpus represented as word IDs
        vector<vector<int>> corpus;
        vector<vector<int>> topics;
        unordered_map<string, int> vocab;
        vector<string> id2word;

        //Estimators and helpers
        vector<vector<double>> theta;
        vector<vector<double>> phi;
        vector<int> nk; 
        vector<int> nm; 

        //Original documents
        vector<Diagnostico> documentos;

        mt19937 generator;

        int K,V,D; //Number of Topics, Vocabulary Size, Number of Documents
        double alpha;
        double beta;
        bool ready = false;
        
    public: 
        Lda(int topics);
        Lda();


        void addDocumentToMatrix(Diagnostico doc);
        void prepareMatrices();
        void firstFilling();
        void secondFilling();
        void run(int iterations);
        void printMatrices();
        void estimators();
        void printTopWordsPerTopic(int topN);
        void printTopDocumentsPerTopic(int topN);
        double logEntropy();
        void calculateEntropy(int k, int iterations);

        void printModel(int topNWords, int topNDocs);

        //Setters and Getters
        void setAlpha(double alpha){ this->alpha = alpha; }
        void setBeta(double beta){ this->beta = beta; }
        double getAlpha() const { return alpha; }
        double getBeta() const { return beta; }
        void setK(int topics){ this->K = topics; }
        int getK() const { return K; }
};

Lda::Lda(int topics){
    this->K = topics;
    K = topics; V = 0; D = 0;
    alpha = double(50)/K;
    beta = .01;
    generator.seed(random_device{}());
}

Lda::Lda(){
    K = 10; V = 0; D = 0;
    alpha = double(50)/K;
    beta = 0.01;
    generator.seed(random_device{}());
}

void Lda::addDocumentToMatrix(Diagnostico doc){
    if(ready){
        errx(1, "No se pueden agregar documentos despues de inicializar el modelo.");
        return;
    }
    vector<string> lemas = doc.getLemas();
    vector<int> wordIds;

    for(const string& palabra : lemas){
        if(vocab.find(palabra) == vocab.end()){
            vocab[palabra] = V;
            id2word.push_back(palabra);
            V++;
        }
        wordIds.push_back(vocab[palabra]);
    }
    corpus.push_back(wordIds);
    documentos.push_back(doc);
    nm.push_back(wordIds.size());
    D++;   
}

void Lda::prepareMatrices(){
    //Inicializar matrices
    //documentsPerTopicMatrix D x K
    documentsPerTopicMatrix.resize(D, vector<int>(K, 0));

    //topicPerWordMatrix K x V
    topicPerWordMatrix.resize(K, vector<int>(V, 0));

    nk.resize(K, 0);

    topics.resize(D);
    theta.resize(D, vector<double>(K, 0.0));
    phi.resize(K, vector<double>(V, 0.0));
    for(int d=0;d<D;d++){   
        topics[d].resize(nm[d]);
    }

    ready = true;
}

void Lda::firstFilling(){
    if(!ready){ errx(1, "Not ready!"); return; }
    for(int d=0;d<D;d++){
        //For each word in document d
        for(int w=0;w<nm[d];w++){
            //Assign a random topic
            uniform_int_distribution<int> distribution(0, K-1);
            int topic = distribution(generator);

            topics[d][w] = topic;
            int wordId = corpus[d][w];

            //Update matrices
            documentsPerTopicMatrix[d][topic]++;
            topicPerWordMatrix[topic][wordId]++;
            nk[topic]++;
            
        }
    }
}

void Lda::secondFilling(){
    if(!ready){ errx(1, "Not ready!"); return; }

    vector<double> probabilities(K, 0.0);

    for(int d=0;d<D;d++){
        for(int w=0;w<nm[d];w++){
            int wordId = corpus[d][w];
            int oldTopic = topics[d][w];
            //Calculate probabilities for each topic
            
            documentsPerTopicMatrix[d][oldTopic]--;
            topicPerWordMatrix[oldTopic][wordId]--;
            nk[oldTopic]--;

            for(int k=0;k<K;k++){
                double wordTopicProb = (topicPerWordMatrix[k][wordId] + beta) / (nk[k] + V * beta);
                double topicDocProb = (documentsPerTopicMatrix[d][k] + alpha ) / (nm[d] + K * alpha);
                probabilities[k] = wordTopicProb * topicDocProb;
            }

            //Sample new topic
            discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
            int newTopic = distribution(generator);

            //Update matrices
            topics[d][w] = newTopic;
            documentsPerTopicMatrix[d][newTopic]++;
            topicPerWordMatrix[newTopic][wordId]++;
            nk[newTopic]++;
        }
    }
}

void Lda::estimators(){
    //Esthimate theta 
    for(int d=0;d<D;d++){
        for(int k=0;k<K;k++){
            theta[d][k] = (documentsPerTopicMatrix[d][k] + alpha) / (nm[d] + K * alpha);
        }
    }

    //Esthimate phi
    for(int k=0;k<K;k++){
        for(int v=0;v<V;v++){
            phi[k][v] = (topicPerWordMatrix[k][v] + beta) / (nk[k] + V * beta);
        }
    }
}

double Lda::logEntropy() {
    if (theta.empty() || phi.empty()) {
        errx(1, "Theta or Phi not computed. Run estimators() first.");
    }

    double totalLog = 0.0;
    long long totalWords = 0;

    for (int d = 0; d < D; d++) {
        totalWords += nm[d];  // total words in document d

        // Count frequencies per word in this document
        unordered_map<int, int> freq;
        for (int w = 0; w < nm[d]; w++) {
            freq[ corpus[d][w] ]++;
        }

        // For each unique word v in document d
        for (auto &entry : freq) {
            int v = entry.first;   // word id
            int n_dv = entry.second;

            // Compute sum_k theta[d][k] * phi[k][v]
            double prob = 0.0;
            for (int k = 0; k < K; k++) {
                prob += theta[d][k] * phi[k][v];
            }

            if (prob > 0)
                totalLog += n_dv * log(prob);
        }
    }

    return - totalLog / double(totalWords);  // normalized log-entropy
}

void Lda::run(int iterations){
    if(!ready){
        errx(1, "Not ready!");
        return;
    }
    firstFilling();
    for(int it=0;it<iterations;it++){
        secondFilling();
        cout << "Iteration " << it+1 << " completed." << endl;
    }
    estimators();
}

void Lda::printModel(int topNWords, int topNDocs){
    printTopWordsPerTopic(topNWords);
    printTopDocumentsPerTopic(topNDocs);
}

void Lda::printMatrices(){
    //save matrices to files
    ofstream docTopicFile("documentsPerTopicMatrix.txt");
    for(int d=0;d<D;d++){
        for(int k=0;k<K;k++){
            docTopicFile << documentsPerTopicMatrix[d][k] << "\t";
        }
        docTopicFile << endl;
    }
    docTopicFile.close();

    ofstream topicWordFile("topicPerWordMatrix.txt");
    for(int k=0;k<K;k++){
        for(int v=0;v<V;v++){
            topicWordFile << topicPerWordMatrix[k][v] << "\t";
        }
        topicWordFile << endl;
    }
    topicWordFile.close();
}

void Lda::printTopWordsPerTopic(int topN){
    ofstream outFile("topWordsPerTopic.txt");
    for(int k = 0; k < K; k++){
        vector<pair<int, double>> wordProbs;

        for(int v = 0; v < V; v++){
            wordProbs.push_back({v, phi[k][v]});
        }

        sort(wordProbs.begin(), wordProbs.end(),
             [](auto &a, auto &b){ return a.second > b.second; });
        outFile << "=========================" << endl; 
        outFile << "Topic " << k << endl;
        for(int i = 0; i < topN && i < wordProbs.size(); i++){
            int wordId = wordProbs[i].first;
            double prob = wordProbs[i].second;
            outFile << "\tWord: " << id2word[wordId] << endl;
            outFile << "\t\tWord ID: " << wordId << endl;
            outFile << "\t\tProbability: " << prob << endl;
        }
    }
    outFile.close();
}


void Lda::printTopDocumentsPerTopic(int topN){
    ofstream outFile("topDocumentsPerTopic.txt");
    for(int k = 0; k < K; k++){
        vector<pair<int, double>> docProbs;

        for(int d = 0; d < D; d++){
            docProbs.push_back({d, theta[d][k]});
        }

        sort(docProbs.begin(), docProbs.end(),
             [](auto &a, auto &b){ return a.second > b.second; });
        
        outFile << "=========================" << endl;
        outFile << "Topic " << k << endl;
        for(int i = 0; i < topN && i < docProbs.size(); i++){
            int docId = docProbs[i].first;
            double prob = docProbs[i].second;
            outFile << "\t Title: " << documentos[docId].getTitulo()<< " " << documentos[docId].getCodigo() << endl;
            outFile << "\t Descripcion: " << documentos[docId].getDefinicion() << endl;
            outFile << "\t\tDocument ID: " << docId << endl;
            outFile << "\t\tProbability: " << prob << endl << endl;
        }
    }
    outFile.close();
}


#include <atomic>
#include <iomanip>

void printProgressBar(int current, int total) {
    float progress = (float)current / total;
    int barWidth = 50;
    int pos = barWidth * progress;

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "%  ";
    std::cout.flush();
}

void Lda::calculateEntropy(int k, int iterations){
    std::cout << "Calculating entropy..." << std::endl;

    std::vector<int> topicsAux;
    for(int i = 1; i < k; i++){
        topicsAux.push_back(i * 5);
    }

    std::vector<double> entropies(topicsAux.size());

    std::atomic<int> completed {0};
    int total = topicsAux.size();

    #pragma omp parallel for
    for(int i = 0; i < topicsAux.size(); i++){
        Lda ldaModel(topicsAux[i]);

        for(int d = 0; d < documentos.size(); d++){
            ldaModel.addDocumentToMatrix(documentos[d]);
        }

        ldaModel.prepareMatrices();
        ldaModel.firstFilling();

        for(int it = 0; it < iterations; it++){
            ldaModel.secondFilling();
        }

        ldaModel.estimators();
        entropies[i] = ldaModel.logEntropy();

        // === actualizar barra de progreso ===
        int done = ++completed;
        #pragma omp critical
        {
            printProgressBar(done, total);
        }
    }

    std::cout << std::endl << "Entropy calculation finished." << std::endl;

    std::ofstream outFile("entropies.csv");
    for(int i = 0; i < topicsAux.size(); i++){
        outFile << topicsAux[i] << "," << entropies[i] << "\n";
    }
}
