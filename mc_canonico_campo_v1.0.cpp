#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <random> // Para una mejor generación de números aleatorios

using namespace std;

// Define constantes
const unsigned long SEED = 0x00001158e460913dL;
const unsigned long MASK48 = 0x00001ffffffffffL;
const int SIDE = 32;
const int Z_SIZE = 2 * SIDE;

// Declaración de Funciones
void monteCarloStep(const char* filename, float upperTemperature, float lowerTemperature, float temperatureStep,
                   float upperField, float lowerField, float fieldStep, int numSteps,
                   float interactionEnergy3NN, float interactionEnergy6NN);
long ranEnt();
double ran0to1();
int ranInt1to8();

// Generador de números aleatorios estático (para mantener el estado)
static unsigned long ale32 = SEED;
static union {
    long tmp;
    float fl;
} xx;

// Función para generar un número aleatorio entero
long ranEnt() {
    ale32 = ale32 * SEED;
    ale32 = ale32 & MASK48;
    return static_cast<long>(ale32);
}

// Función para generar un número aleatorio entre 0 y 1 (inclusive)
double ran0to1() {
    xx.tmp = ranEnt();
    xx.tmp = (xx.tmp & 0x3fffffffL) | 0x3f800000l;
    return static_cast<double>(xx.fl - 1.0);
}

// Función para generar un número aleatorio entero entre 1 y 8 (inclusive)
int ranInt1to8() {
    xx.fl = ranEnt();
    xx.fl = static_cast<float>(xx.tmp & 0x00000007L);
    return static_cast<int>(xx.fl + 1.0);
}

int main() {
    clock_t start = clock(); // Iniciamos el reloj

    const int NUM_STEPS = 100000;
    const float INTERACTION_J3 = 150.0; // Energía de interacción para vecinos de tercer orden
    const float INTERACTION_J6 = 100.0; // Energía de interacción para vecinos de sexto orden
    const float TEMP_UPPER = 300.0;
    const float TEMP_LOWER = 300.0;
    const float TEMP_STEP = 10.0;
    const float FIELD_UPPER = 200.0;
    const float FIELD_LOWER = -200.0;
    const float FIELD_STEP = 20.0;

    vector<string> filenames = {"cu-al-mn_0.67-0.25-0.08"};

    for (const auto& filename : filenames) {
        cout << "Archivo entrada: " << filename << endl;
        monteCarloStep(filename.c_str(), TEMP_UPPER, TEMP_LOWER, TEMP_STEP,
                       FIELD_UPPER, FIELD_LOWER, FIELD_STEP, NUM_STEPS,
                       INTERACTION_J3, INTERACTION_J6);
    }

    clock_t end = clock(); // Cortamos el reloj
    double timeTaken = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    // Display the time taken in the format "hh:mm:ss"
    int hours = static_cast<int>(timeTaken / 3600);
    int minutes = static_cast<int>((timeTaken - hours * 3600) / 60);
    int seconds = static_cast<int>(timeTaken - hours * 3600 - minutes * 60);
    cout << "Tiempo total: "
         << setfill('0') << setw(2) << hours << ":"
         << setfill('0') << setw(2) << minutes << ":"
         << setfill('0') << setw(2) << seconds << endl;

    return 0;
}

// Función Paso MC
void monteCarloStep(const char* filename, float upperTemperature, float lowerTemperature, float temperatureStep,
                   float upperField, float lowerField, float fieldStep, int numSteps,
                   float interactionEnergy3NN, float interactionEnergy6NN) {

    string initialConfigFilename = string(filename) + "_365.txt";
    ifstream inputFile(initialConfigFilename, ios::in | ios::binary);
    if (!inputFile) {
        cerr << "No se puede abrir el archivo de entrada: " << initialConfigFilename << endl;
        return;
    }

    // ABRIÓ ARCHIVO DE LA RED INICIAL
    float lattice[SIDE][SIDE][Z_SIZE];
    float magneticMoment[SIDE][SIDE][Z_SIZE];
    float aux;

    for (int i = 0; i < SIDE; ++i) {
        for (int j = 0; j < SIDE; ++j) {
            for (int k = 0; k < Z_SIZE; ++k) {
                inputFile >> aux;
                lattice[i][j][k] = (-1.0f / 12.0f) * aux * aux * aux + (1.0f / 3.0f) * aux * aux + (7.0f / 12.0f) * aux - (5.0f / 6.0f);
                magneticMoment[i][j][k] = (-1.0f / 12.0f) * aux * aux * aux - (1.0f / 3.0f) * aux * aux + (7.0f / 12.0f) * aux + (5.0f / 6.0f);
            }
        }
    }
    inputFile.close();
    // CERRÓ ARCHIVO DE LA RED INICIAL

    string outputFilename = string(filename) + "_out.txt";
    ofstream outputFile(outputFilename, ios::out | ios::app);
    if (!outputFile) {
        cerr << "No se puede abrir el archivo de parametros LRO: " << outputFilename << endl;
        return;
    }

    // Escribo la lista de campos para recorrer con el for loop
    vector<float> fieldList;
    for (float i = 0; i <= upperField; i += fieldStep) {
        fieldList.push_back(i);
    }
    for (float i = upperField; i >= lowerField; i -= fieldStep) {
        fieldList.push_back(i);
    }
    for (float i = lowerField; i <= upperField; i += fieldStep) {
        fieldList.push_back(i);
    }
    // Termina la lista de campos

    cout << "J3=" << interactionEnergy3NN << "; J6=" << interactionEnergy6NN << endl;

    for (float temperature = lowerTemperature; temperature <= upperTemperature; temperature += temperatureStep) {
        cout << endl << "Trabajando a T = " << temperature << endl;
        int fieldCounter = 0;
        for (float magneticField : fieldList) {
            cout << endl << "Trabajando a H = " << magneticField << endl;

            float deltaEnergyAccumulatedMagnetic = 0.0f;

            for (int step = 1; step <= numSteps; ++step) {
                for (int x = 0; x < SIDE; ++x) {
                    for (int y = 0; y < SIDE; ++y) {
                        for (int z = 0; z < Z_SIZE; ++z) {
                            float currentSpin = magneticMoment[x][y][z];

                            // Condiciones de contorno periódicas. Coordenadas de los vecinos
                            int infX, supX, infY, supY;
                            if (abs(fmod(z, 2.0f)) > 1e-6) { // Si Z es impar
                                infX = x;
                                supX = (x + 1) % SIDE;
                                infY = y;
                                supY = (y + 1) % SIDE;
                            } else { // Si Z es par
                                infX = (x - 1 + SIDE) % SIDE;
                                supX = x;
                                infY = (y - 1 + SIDE) % SIDE;
                                supY = y;
                            }
                            int infZ = (z - 1 + Z_SIZE) % Z_SIZE;
                            int supZ = (z + 1) % Z_SIZE;

                            // 2dos. Vecinos
                            int infX2 = (x - 1 + SIDE) % SIDE;
                            int supX2 = (x + 1) % SIDE;
                            int infY2 = (y - 1 + SIDE) % SIDE;
                            int supY2 = (y + 1) % SIDE;
                            int infZ2 = (z - 2 + Z_SIZE) % Z_SIZE;
                            int supZ2 = (z + 2) % Z_SIZE;

                            // 6tos. Vecinos
                            int infX6 = (x - 2 + SIDE) % SIDE;
                            int supX6 = (x + 2) % SIDE;
                            int infY6 = (y - 2 + SIDE) % SIDE;
                            int supY6 = (y + 2) % SIDE;
                            int infZ6 = (z - 4 + Z_SIZE) % Z_SIZE;
                            int supZ6 = (z + 4) % Z_SIZE;

                            // Sumas para calcular diferencias de energías magnéticas
                            float sumNearestNeighbors =
                                magneticMoment[infX][infY][infZ] + magneticMoment[supX][infY][infZ] +
                                magneticMoment[infX][supY][infZ] + magneticMoment[infX][infY][supZ] +
                                magneticMoment[supX][supY][infZ] + magneticMoment[supX][infY][supZ] +
                                magneticMoment[infX][supY][supZ] + magneticMoment[supX][supY][supZ];

                            float sumNextNearestNeighbors =
                                magneticMoment[infX2][y][z] + magneticMoment[supX2][y][z] +
                                magneticMoment[x][infY2][z] + magneticMoment[x][supY2][z] +
                                magneticMoment[x][y][infZ2] + magneticMoment[x][y][supZ2];

                            float sumThirdNearestNeighbors =
                                magneticMoment[supX2][supY2][z] + magneticMoment[supX2][infY2][z] +
                                magneticMoment[infX2][supY2][z] + magneticMoment[infX2][infY2][z] +
                                magneticMoment[x][supY2][supZ2] + magneticMoment[x][infY2][supZ2] +
                                magneticMoment[x][supY2][infZ2] + magneticMoment[x][infY2][infZ2] +
                                magneticMoment[supX2][y][supZ2] + magneticMoment[infX2][y][supZ2] +
                                magneticMoment[supX2][y][infZ2] + magneticMoment[infX2][y][infZ2];

                            float sumSixthNearestNeighbors =
                                magneticMoment[supX6][y][z] + magneticMoment[infX6][y][z] +
                                magneticMoment[x][supY6][z] + magneticMoment[x][infY6][z] +
                                magneticMoment[x][y][supZ6] + magneticMoment[x][y][infZ6];

                            // Metropolis Magnético
                            if (abs(currentSpin) > 1e-6) {
                                float deltaEnergyMagneticInteraction =
                                    2.0f * 0.0f * currentSpin * sumNearestNeighbors + // J1 = 0 por ahora
                                    2.0f * 0.0f * currentSpin * sumNextNearestNeighbors + // J2 = 0 por ahora
                                    2.0f * interactionEnergy3NN * currentSpin * sumThirdNearestNeighbors +
                                    2.0f * interactionEnergy6NN * currentSpin * sumSixthNearestNeighbors;

                                float deltaEnergyExternalField = 2.0f * magneticField * currentSpin;
                                float deltaEnergy = deltaEnergyMagneticInteraction + deltaEnergyExternalField;

                                if (deltaEnergy <= 0.0f) {
                                    magneticMoment[x][y][z] = -currentSpin;
                                    deltaEnergyAccumulatedMagnetic += deltaEnergy;
                                } else {
                                    double epsilon = ran0to1();
                                    float boltzmannFactor = exp(-deltaEnergy / temperature);
                                    if (boltzmannFactor >= epsilon) {
                                        magneticMoment[x][y][z] = -currentSpin;
                                        deltaEnergyAccumulatedMagnetic += deltaEnergy;
                                    }
                                }
                            } // Fin del Metropolis Magnético
                        } // Fin del for Z
                    } // Fin del for Y
                } // Fin del for X

                // CÁLCULO DE LOS PARÁMETROS DE LRO
                if (step > (numSteps - 200)) {
                    int cuI = 0, cuII = 0, cuIII = 0, cuIV = 0;
                    int mnUpI = 0, mnUpII = 0, mnUpIII = 0, mnUpIV = 0, mnDownI = 0, mnDownII = 0, mnDownIII = 0, mnDownIV = 0;
                    int alI = 0, alII = 0, alIII = 0, alIV = 0;
                    long magnetization = 0;

                    for (int i = 0; i < SIDE; ++i) {
                        for (int j = 0; j < SIDE; ++j) {
                            for (int k = 0; k < Z_SIZE; ++k) {
                                magnetization += static_cast<long>(magneticMoment[i][j][k]);

                                if (k % 2 == 0) { // Subred I y II
                                    if ((i + j + (k / 2)) % 2 == 0) { // Subred I
                                        if (abs(1.0f - lattice[i][j][k]) < 1e-6) {
                                            cuI++;
                                        }
                                        if (abs(0.0f - lattice[i][j][k]) < 1e-6) {
                                            if (magneticMoment[i][j][k] > 1e-6) {
                                                mnUpI++;
                                            } else {
                                                mnDownI++;
                                            }
                                        }
                                        if (abs(-1.0f - lattice[i][j][k]) < 1e-6) {
                                            alI++;
                                        }
                                    } else { // Subred II
                                        if (abs(1.0f - lattice[i][j][k]) < 1e-6) {
                                            cuII++;
                                        }
                                        if (abs(0.0f - lattice[i][j][k]) < 1e-6) {
                                            if (magneticMoment[i][j][k] > 1e-6) {
                                                mnUpII++;
                                            } else {
                                                mnDownII++;
                                            }
                                        }
                                        if (abs(-1.0f - lattice[i][j][k]) < 1e-6) {
                                            alII++;
                                        }
                                    }
                                } else { // Subred III, IV
                                    if ((i + j + (k / 2)) % 2 == 0) { // Subred III
                                        if (abs(1.0f - lattice[i][j][k]) < 1e-6) {
                                            cuIII++;
                                        }
                                        if (abs(0.0f - lattice[i][j][k]) < 1e-6) {
                                            if (magneticMoment[i][j][k] > 1e-6) {
                                                mnUpIII++;
                                            } else {
                                                mnDownIII++;
                                            }
                                        }
                                        if (abs(-1.0f - lattice[i][j][k]) < 1e-6) {
                                            alIII++;
                                        }
                                    } else { // Subred IV
                                        if (abs(1.0f - lattice[i][j][k]) < 1e-6) {
                                            cuIV++;
                                        }
                                        if (abs(0.0f - lattice[i][j][k]) < 1e-6) {
                                            if (magneticMoment[i][j][k] > 1e-6) {
                                                mnUpIV++;
                                            } else {
                                                mnDownIV++;
                                            }
                                        }
                                        if (abs(-1.0f - lattice[i][j][k]) < 1e-6) {
                                            alIV++;
                                        }
                                    }
                                }
                            } // final del for k
                        } //final del for j
                    } //final del for i

                    float totalSites = static_cast<float>(SIDE * SIDE * Z_SIZE);
                    float xCu = static_cast<float>(cuI + cuII - cuIII - cuIV) / totalSites;
                    float xMnUp = static_cast<float>(mnUpI + mnUpII - mnUpIII - mnUpIV) / totalSites;
                    float xMnDown = static_cast<float>(mnDownI + mnDownII - mnDownIII - mnDownIV) / totalSites;
                    float xAl = static_cast<float>(alI + alII - alIII - alIV) / totalSites;
                    float yCu = static_cast<float>(cuI - cuII) * 2.0f / totalSites;
                    float yMnUp = static_cast<float>(mnUpI - mnUpII) * 2.0f / totalSites;
                    float yMnDown = static_cast<float>(mnDownI - mnDownII) * 2.0f / totalSites;
                    float yAl = static_cast<float>(alI - alII) * 2.0f / totalSites;
                    float zCu = static_cast<float>(cuIII - cuIV) * 2.0f / totalSites;
                    float zMnUp = static_cast<float>(mnUpIII - mnUpIV) * 2.0f / totalSites;
                    float zMnDown = static_cast<float>(mnDownIII - mnDownIV) * 2.0f / totalSites;
                    float zAl = static_cast<float>(alIII - alIV) * 2.0f / totalSites;
                    outputFile << step << "\t";
                    outputFile << magneticField << "\t";
                    outputFile << temperature << "\t";
                    outputFile << xCu << "\t";
                    outputFile << xMnUp << "\t";
                    outputFile << xMnDown << "\t";
                    outputFile << xAl << "\t";
                    outputFile << yCu << "\t";
                    outputFile << yMnUp << "\t";
                    outputFile << yMnDown << "\t";
                    outputFile << yAl << "\t";
                    outputFile << zCu << "\t";
                    outputFile << zMnUp << "\t";
                    outputFile << zMnDown << "\t";
                    outputFile << zAl << "\t";
                    outputFile << static_cast<float>(magnetization) / totalSites << "\t";
                    outputFile << deltaEnergyAccumulatedMagnetic << "\t" << endl;
                } // terminó con los imprimir los pasos
            }     // Terminó con el número de pasos pedido

            string finalConfigFilename = string(filename) + "_" + to_string(static_cast<int>(magneticField)) + "_" + to_string(fieldCounter) + ".txt";
            ofstream outputLatticeFile(finalConfigFilename, ios::out); // ABRE ARCHIVO RED

            if (!outputLatticeFile) {
                cerr << "No se pudo abrir el archivo de salida de la red: " << finalConfigFilename << endl;
                return;
            }

            for (int i = 0; i < SIDE; ++i) {
                for (int j = 0; j < SIDE; ++j) {
                    for (int k = 0; k < Z_SIZE; ++k) {
                        outputLatticeFile << (0.5f * lattice[i][j][k] * lattice[i][j][k] + 1.5f * lattice[i][j][k] - 0.5f * magneticMoment[i][j][k] * magneticMoment[i][j][k] + 1.5f * magneticMoment[i][j][k]) << endl;
                        // VUELCA MATRIZ
                    }
                }
            }
            outputLatticeFile.close(); // CIERRA ARCHIVO DE SALIDA

            fieldCounter++; // incremento el contador
        }                  // Terminó con la variación de campo
    }                      // Terminó con la variación de temperatura

    outputFile.close(); /// CERRÓ ARCHIVO DE PARÁMETROS LRO

    cout << "***********" << endl << "*TERMINADO*" << endl << "***********" << endl;
} // Final del VOID monteCarloStep