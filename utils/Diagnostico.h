#include <iostream>
#include <vector>

using namespace std;

class Diagnostico{
    private:
        string titulo;
        string codigo;
        string definicion;
        vector<string> lemas;

    public:
        Diagnostico(string titulo, string codigo,
                    string definicion, vector<string> lemas);
        Diagnostico();
        void setTitulo(const string& titulo);
        void setCodigo(const string& codigo);
        void setDefinicion(const string& definicion);
        void setLemas(const vector<string>& lemas);

        string getTitulo() const;
        string getCodigo() const;
        string getDefinicion() const;
        vector<string> getLemas() const;
};

Diagnostico::Diagnostico(string titulo, string codigo,
                         string definicion, vector<string> lemas)
    : titulo(titulo), codigo(codigo), definicion(definicion), lemas(lemas) {}

Diagnostico::Diagnostico(){}

void Diagnostico::setTitulo(const string& titulo) {
    this->titulo = titulo;
}

void Diagnostico::setCodigo(const string& codigo) {
    this->codigo = codigo;
}

void Diagnostico::setDefinicion(const string& definicion) {
    this->definicion = definicion;
}

void Diagnostico::setLemas(const vector<string>& lemas) {
    this->lemas = lemas;
}

string Diagnostico::getTitulo() const {
    return titulo;
}

string Diagnostico::getCodigo() const {
    return codigo;
}

string Diagnostico::getDefinicion() const {
    return definicion;
}

vector<string> Diagnostico::getLemas() const {
    return lemas;
}