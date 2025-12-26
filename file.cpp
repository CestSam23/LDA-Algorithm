#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include "json.hpp"
#include "utils/Lda.h"


using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace std;


bool debug = true;

// --- Leer un solo JSON ---
Diagnostico leerDiagnostico(const fs::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir: " + path.string());
    }

    json data;
    file >> data;

    Diagnostico d;
    d.setTitulo(data.value("titulo", ""));
    d.setCodigo(data.value("codigo", ""));
    d.setDefinicion(data.value("definicion", ""));
    d.setLemas(data.value("lemas", std::vector<std::string>{}));
    return d;
}

// --- Leer todos los JSON en los subdirectorios Dominio1...Dominio13 ---
std::vector<std::vector<Diagnostico>> leerCorpus(const std::string& rutaBase) {

    // Vector corpus
    std::vector<std::vector<Diagnostico>> corpus;
    
    //Iterador para ruta base.
    for (const auto& dominio : fs::directory_iterator(rutaBase)) {
        //Ignorar cualquier otro documento que no sea carpeta
        if (!dominio.is_directory()) continue;

        std::vector<Diagnostico> documentosEnDominio;


        //Itera sobre los archivos dentro de las carpetas dominio
        for (const auto& archivo : fs::directory_iterator(dominio.path())) {
            
            //Itera sobre los json, trata de abrirlos
            if (archivo.path().extension() == ".json") {
                try {
                    Diagnostico d = leerDiagnostico(archivo.path());
                    documentosEnDominio.push_back(d);
                } catch (const std::exception& e) {
                    std::cerr << "Error en " << archivo.path() << ": " << e.what() << "\n";
                }
            }
        }

        //Agregar si no esta vacio
        if(!documentosEnDominio.empty()){
            corpus.push_back(documentosEnDominio);
        }
    }

    return corpus;
}

int main() {
    try {
        std::string rutaCorpus = "../Lematizador/Corpus";
        std::vector<std::vector<Diagnostico>> corpus = leerCorpus(rutaCorpus);

        Lda ldaModel(250); 


        for(int d=0;d<corpus.size();d++){
            for(int j=0;j<corpus[d].size();j++){
                ldaModel.addDocumentToMatrix(corpus[d][j]);
            }
        }

        ldaModel.prepareMatrices();
        ldaModel.run(1000);
        ldaModel.printModel(20,20);
        ldaModel.calculateEntropy(100, 100);

        
    }
    catch (const std::exception& e) {
        std::cerr << "Error general: " << e.what() << "\n";
    }
}


