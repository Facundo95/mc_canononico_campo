#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <tuple>
#include <random>
#include <chrono>

// Forward declarations
class Lattice;
class MonteCarloSimulation;
class OrderParameterCalculator;
class OutputManager;
class TimeCalculator;

// Constants
const unsigned long SEED = 0x00001158e460913dL;
const unsigned long MASK48 = 0x00001ffffffffffL;
const int SIDE_LENGTH = 32;

// Random number generator (moved to a utility function or kept static)
static unsigned long ale32 = SEED;
static union { long tmp; float fl; } xx;
long ranEnt() { ale32 = ale32 * SEED; ale32 = ale32 & MASK48; return static_cast<long>(ale32); }
double ran0to1() { xx.tmp = ranEnt(); xx.tmp = (xx.tmp & 0x3fffffffL) | 0x3f800000l; return static_cast<double>(xx.fl - 1.0); }

class Lattice {
private:
    std::vector<std::vector<std::vector<float>>> latticeConfig;
    std::vector<std::vector<std::vector<float>>> magneticMoments;
    int side;
    int zSize;

public:
    Lattice(const std::string& filename) : side(SIDE_LENGTH), zSize(2 * SIDE_LENGTH),
                                            latticeConfig(side, std::vector<std::vector<float>>(side, std::vector<float>(zSize))),
                                            magneticMoments(side, std::vector<std::vector<float>>(side, std::vector<float>(zSize))) {
        loadFromFile(filename);
    }

    int getSideLength() const { return side; }
    int getZSize() const { return zSize; }
    float getLatticeValue(int x, int y, int z) const { return latticeConfig[x][y][z]; }
    float getMagneticMoment(int x, int y, int z) const { return magneticMoments[x][y][z]; }
    void setMagneticMoment(int x, int y, int z, float value) { magneticMoments[x][y][z] = value; }

    std::tuple<int, int, int> getNeighborIndices(int x, int y, int z, int dx, int dy, int dz) const {
        int nx = (x + dx + side) % side;
        int ny = (y + dy + side) % side;
        int nz = (z + dz + zSize) % zSize;
        return std::make_tuple(nx, ny, nz);
    }

private:
    void loadFromFile(const std::string& filename) {
        std::string initialConfigFilename = filename + "_365.txt";
        std::ifstream inputFile(initialConfigFilename, std::ios::in | std::ios::binary);
        if (!inputFile) {
            std::cerr << "Error: Could not open input file " << initialConfigFilename << std::endl;
            exit(1);
        }
        float aux;
        for (int i = 0; i < side; ++i) {
            for (int j = 0; j < side; ++j) {
                for (int k = 0; k < zSize; ++k) {
                    inputFile >> aux;
                    latticeConfig[i][j][k] = (-1.0f / 12.0f) * aux * aux * aux + (1.0f / 3.0f) * aux * aux + (7.0f / 12.0f) * aux - (5.0f / 6.0f);
                    magneticMoments[i][j][k] = (-1.0f / 12.0f) * aux * aux * aux - (1.0f / 3.0f) * aux * aux + (7.0f / 12.0f) * aux + (5.0f / 6.0f);
                }
            }
        }
        inputFile.close();
    }
};

class MonteCarloSimulation {
private:
    float temperature;
    float magneticField;
    int steps;
    float j3;
    float j6;
    Lattice& lattice;

public:
    MonteCarloSimulation(Lattice& lat, int numSteps, float j3_val, float j6_val)
        : lattice(lat), steps(numSteps), j3(j3_val), j6(j6_val), temperature(0.0f), magneticField(0.0f) {}

    void setTemperature(float temp) { temperature = temp; }
    void setMagneticField(float field) { magneticField = field; }

    float performMCSweep(float& deltaEnergyAccumulated) {
        deltaEnergyAccumulated = 0.0f;
        int side = lattice.getSideLength();
        int zSize = lattice.getZSize();
        for (int x = 0; x < side; ++x) {
            for (int y = 0; y < side; ++y) {
                for (int z = 0; z < zSize; ++z) {
                    float currentSpin = lattice.getMagneticMoment(x, y, z);
                    if (std::abs(currentSpin) > 1e-6) {
                        float deltaE = calculateDeltaEnergy(x, y, z, currentSpin);
                        if (deltaE <= 0.0f) {
                            lattice.setMagneticMoment(x, y, z, -currentSpin);
                            deltaEnergyAccumulated += deltaE;
                        } else {
                            if (ran0to1() < std::exp(-deltaE / temperature)) {
                                lattice.setMagneticMoment(x, y, z, -currentSpin);
                                deltaEnergyAccumulated += deltaE;
                            }
                        }
                    }
                }
            }
        }
        return deltaEnergyAccumulated;
    }

private:
    float calculateDeltaEnergy(int x, int y, int z, float currentSpin) const {
        int side = lattice.getSideLength();
        int zSize = lattice.getZSize();

        auto neighbor = [&](int dx, int dy, int dz) {
            return lattice.getMagneticMoment(std::get(lattice.getNeighborIndices(x, y, z, dx, dy, dz)),
                                             std::get(lattice.getNeighborIndices(x, y, z, dx, dy, dz)),
                                             std::get(lattice.getNeighborIndices(x, y, z, dx, dy, dz)));
        };

        float sumNearestNeighbors = 0.0f;
        if (std::abs(std::fmod(z, 2.0f)) > 1e-6) { // Z is odd
            sumNearestNeighbors += neighbor(1, 0, 1) + neighbor(-1, 0, 1) + neighbor(0, 1, 1) + neighbor(0, -1, 1) +
                                   neighbor(1, 0, -1) + neighbor(-1, 0, -1) + neighbor(0, 1, -1) + neighbor(0, -1, -1);
        } else { // Z is even
            sumNearestNeighbors += neighbor(1, 0, 1) + neighbor(-1, 0, 1) + neighbor(0, 1, 1) + neighbor(0, -1, 1) +
                                   neighbor(1, 0, -1) + neighbor(-1, 0, -1) + neighbor(0, 1, -1) + neighbor(0, -1, -1);
        }
        sumNearestNeighbors += neighbor(1, 0, 0) + neighbor(-1, 0, 0) + neighbor(0, 1, 0) + neighbor(0, -1, 0); // In-plane

        float sumNextNearestNeighbors = neighbor(1, 0, 0) + neighbor(-1, 0, 0) + neighbor(0, 1, 0) + neighbor(0, -1, 0) + neighbor(0, 0, 2) + neighbor(0, 0, -2);
        float sumThirdNearestNeighbors = 0.0f; // Implement based on your neighbor definition
        float sumSixthNearestNeighbors = 0.0f; // Implement based on your neighbor definition

        float deltaEnergyMagneticInteraction =
            2.0f * 0.0f * currentSpin * sumNearestNeighbors +
            2.0f * 0.0f * currentSpin * sumNextNearestNeighbors +
            2.0f * j3 * currentSpin * sumThirdNearestNeighbors +
            2.0f * j6 * currentSpin * sumSixthNearestNeighbors;

        float deltaEnergyExternalField = 2.0f * magneticField * currentSpin;

        return deltaEnergyMagneticInteraction + deltaEnergyExternalField;
    }
};

class OrderParameterCalculator {
private:
    const Lattice& lattice;

public:
    OrderParameterCalculator(const Lattice& lat) : lattice(lat) {}

    std::tuple<float, float, float, float, float, float, float, float, float, float, float, float, float> calculateLROParameters() const {
        int side = lattice.getSideLength();
        int zSize = lattice.getZSize();
        int cuI = 0, cuII = 0, cuIII = 0, cuIV = 0;
        int mnUpI = 0, mnUpII = 0, mnUpIII = 0, mnUpIV = 0, mnDownI = 0, mnDownII = 0, mnDownIII = 0, mnDownIV = 0;
        int alI = 0, alII = 0, alIII = 0, alIV = 0;

        for (int i = 0; i < side; ++i) {
            for (int j = 0; j < side; ++j) {
                for (int k = 0; k < zSize; ++k) {
                    float latticeValue = lattice.getLatticeValue(i, j, k);
                    float magneticMoment = lattice.getMagneticMoment(i, j, k);

                    if (k % 2 == 0) {
                        if ((i + j + (k / 2)) % 2 == 0) {
                            if (std::abs(1.0f - latticeValue) < 1e-6) cuI++;
                            if (std::abs(0.0f - latticeValue) < 1e-6) (magneticMoment > 1e-6) ? mnUpI++ : mnDownI++;
                            if (std::abs(-1.0f - latticeValue) < 1e-6) alI++;
                        } else {
                            if (std::abs(1.0f - latticeValue) < 1e-6) cuII++;
                            if (std::abs(0.0f - latticeValue) < 1e-6) (magneticMoment > 1e-6) ? mnUpII++ : mnDownII++;
                            if (std::abs(-1.0f - latticeValue) < 1e-6) alII++;
                        }
                    } else {
                        if ((i + j + (k / 2)) % 2 == 0) {
                            if (std::abs(1.0f - latticeValue) < 1e-6) cuIII++;
                            if (std::abs(0.0f - latticeValue) < 1e-6) (magneticMoment > 1e-6) ? mnUpIII++ : mnDownIII++;
                            if (std::abs(-1.0f - latticeValue) < 1e-6) alIII++;
                        } else {
                            if (std::abs(1.0f - latticeValue) < 1e-6) cuIV++;
                            if (std::abs(0.0f - latticeValue) < 1e-6) (magneticMoment > 1e-6) ? mnUpIV++ : mnDownIV++;
                            if (std::abs(-1.0f - latticeValue) < 1e-6) alIV++;
                        }
                    }
                }
            }
        }

        float totalSites = static_cast<float>(side * side * zSize);
        return std::make_tuple(
            static_cast<float>(cuI + cuII - cuIII - cuIV) / totalSites,
            static_cast<float>(mnUpI + mnUpII - mnUpIII - mnUpIV) / totalSites,
            static_cast<float>(mnDownI + mnDownII - mnDownIII - mnDownIV) / totalSites,
            static_cast<float>(alI + alII - alIII - alIV) / totalSites,
            static_cast<float>(cuI - cuII) * 2.0f / totalSites,
            static_cast<float>(mnUpI - mnUpII) * 2.0f / totalSites,
            static_cast<float>(mnDownI - mnDownII) * 2.0f / totalSites,
            static_cast<float>(alI - alII) * 2.0f / totalSites,
            static_cast<float>(cuIII - cuIV) * 2.0f / totalSites,
            static_cast<float>(mnUpIII - mnUpIV) * 2.0f / totalSites,
            static_cast<float>(mnDownIII - mnDownIV) * 2.0f / totalSites,
            static_cast<float>(alIII - alIV) * 2.0f / totalSites
        );
    }
    
    float calculateAverageMagnetization() const {
        int side = lattice.getSideLength();
        int zSize = lattice.getZSize();
        long magnetization = 0;
        for (int i = 0; i < side; ++i) {
            for (int j = 0; j < side; ++j) {
                for (int k = 0; k < zSize; ++k) {
                    magnetization += static_cast<long>(lattice.getMagneticMoment(i, j, k));
                }
            }
        }
        return static_cast<float>(magnetization) / (side * side * zSize);
    }
};
    
class OutputManager {
private:
    std::string baseFilename;
    std::ofstream lroFileStream;
    
public:
    OutputManager(const std::string& baseName) : baseFilename(baseName) {
        lroFileStream.open(baseFilename + "_out.txt", std::ios::out | std::ios::app);
        if (!lroFileStream.is_open()) {
            std::cerr << "Error: Could not open LRO output file." << std::endl;
            exit(1);
        }
    }

    void writeLROData(int step, float field, float temp,
                        float xCu, float xMnUp, float xMnDown, float xAl,
                        float yCu, float yMnUp, float yMnDown, float yAl,
                        float zCu, float zMnUp, float zMnDown, float zAl,
                        float magnetization, float deltaEnergy) {
        lroFileStream << step << "\t" << field << "\t" << temp << "\t"
                    << xCu << "\t" << xMnUp << "\t" << xMnDown << "\t" << xAl << "\t"
                    << yCu << "\t" << yMnUp << "\t" << yMnDown << "\t" << yAl << "\t"
                    << zCu << "\t" << zMnUp << "\t" << zMnDown << "\t" << zAl << "\t"
                    << magnetization << "\t" << deltaEnergy << "\t" << std::endl;
    }
    
    void writeFinalConfiguration(const Lattice& lattice, float field, int counter) {
        std::string filename = baseFilename + "_" + std::to_string(static_cast<int>(field)) + "_" + std::to_string(counter) + ".txt";
        std::ofstream configFile(filename, std::ios::out);
        if (!configFile.is_open()) {
            std::cerr << "Error: Could not open configuration output file." << std::endl;
            return;
        }
        int side = lattice.getSideLength();
        int zSize = lattice.getZSize();
        for (int i = 0; i < side; ++i) {
            for (int j = 0; j < side; ++j) {
                for (int k = 0; k < zSize; ++k) {
                    configFile << (0.5f * lattice.getLatticeValue(i, j, k) * lattice.getLatticeValue(i, j, k) + 1.5f * lattice.getLatticeValue(i, j, k) -
                                   0.5f * lattice.getMagneticMoment(i, j, k) * lattice.getMagneticMoment(i, j, k) + 1.5f * lattice.getMagneticMoment(i, j, k))
                               << std::endl;
                }
            }
        }
        configFile.close();
    }
    
    ~OutputManager() {
        if (lroFileStream.is_open()) {
            lroFileStream.close();
        }
    }
};

class TimeCalculator {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::chrono::time_point<std::chrono::high_resolution_clock> endTime;
    double durationSeconds;
 
public:
    TimeCalculator() : durationSeconds(0.0) {}

    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }
 
    void stop() {
        endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;
        durationSeconds = duration.count();
    }
    
    double getDurationSeconds() const {
        return durationSeconds;
    }
    
    void displayDuration() const {
        int hours = static_cast<int>(durationSeconds / 3600);
        int minutes = static_cast<int>((durationSeconds - hours * 3600) / 60);
        int seconds = static_cast<int>(durationSeconds - hours * 3600 - minutes * 60);
        std::cout << "Total time: "
        << std::setfill('0') << std::setw(2) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds << std::endl;
    }
};

int main() {
    TimeCalculator timer;
    timer.start();

    const int NUM_STEPS = 100000;
    const float INTERACTION_J3 = 150.0f;
    const float INTERACTION_J6 = 100.0f;
    const float TEMP_UPPER = 300.0f;
    const float TEMP_LOWER = 300.0f;
    const float TEMP_STEP = 10.0f;
    const float FIELD_UPPER = 200.0f;
    const float FIELD_LOWER = -200.0f;
    const float FIELD_STEP = 20.0f;
    
    std::vector<std::string> filenames = {"cu-al-mn_0.67-0.25-0.08"};
    
    for (const auto& filename : filenames) {
        std::cout << "Processing file: " << filename << std::endl;
        Lattice lattice(filename);
        MonteCarloSimulation mcSimulation(lattice, NUM_STEPS, INTERACTION_J3, INTERACTION_J6);
        OrderParameterCalculator opCalculator(lattice);
        OutputManager outputManager(filename);
        
        for (float temp = TEMP_LOWER; temp <= TEMP_UPPER; temp += TEMP_STEP) {
            std::cout << "\nWorking at T = " << temp << std::endl;
            mcSimulation.setTemperature(temp);
            int fieldCounter = 0;
            for (float field : {0.0f, 20.0f, 40.0f, 60.0f, 80.0f, 100.0f, 120.0f, 140.0f, 160.0f, 180.0f, 200.0f,
                180.0f, 160.0f, 140.0f, 120.0f, 100.0f, 80.0f, 60.0f, 40.0f, 20.0f, 0.0f,
                -20.0f, -40.0f, -60.0f, -80.0f, -100.0f, -120.0f, -140.0f, -160.0f, -180.0f, -200.0f,
                -180.0f, -160.0f, -140.0f, -120.0f, -100.0f, -80.0f, -60.0f, -40.0f, -20.0f, 0.0f}) {
                    std::cout << " Working at H = " << field << std::endl;
                    mcSimulation.setMagneticField(field);
                    float deltaEnergyAccumulated = 0.0f;
                    for (int step = 1; step <= NUM_STEPS; ++step) {
                        mcSimulation.performMCSweep(deltaEnergyAccumulated);
                        if (step > (NUM_STEPS - 200)) {
                            auto [xCu, xMnUp, xMnDown, xAl, yCu, yMnUp, yMnDown, yAl, zCu, zMnUp, zMnDown, zAl] = opCalculator.calculateLROParameters();
                            float magnetization = opCalculator.calculateAverageMagnetization();
                            outputManager.writeLROData(step, field, temp,
                                xCu, xMnUp, xMnDown, xAl,
                                yCu, yMnUp, yMnDown, yAl,
                                zCu, zMnUp, zMnDown, zAl,
                                magnetization, deltaEnergyAccumulated);
                            }
                        }
                        outputManager.writeFinalConfiguration(lattice, field, fieldCounter++);
                    }
                }
            }
            
            timer.stop();
            timer.displayDuration();
            
            return 0;
    }