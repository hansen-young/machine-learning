#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include "autograd.h"

struct Data {
    double x0;
    double x1;
    double y;
};

std::vector<Data> readCSV(const std::string& filename) {
    std::vector<Data> data;
    std::ifstream file(filename);
    std::string line;

    if (file.is_open()) {
        std::getline(file, line);  // Skip the header

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            Data row;

            std::getline(ss, value, ',');
            row.x0 = std::stod(value);

            std::getline(ss, value, ',');
            row.x1 = std::stod(value);

            std::getline(ss, value, ',');
            row.y = std::stod(value);

            data.push_back(row);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }

    return data;
}

int main() {
    std::vector<Data> dataset = readCSV("src/data.csv");

    int epoch = 150;
    double learningRate = 0.01;

    auto x0 = autograd::createValue(50, true);
    auto x1 = autograd::createValue(50, true);
    auto b = autograd::createValue(0, true);

    for (int i = 0; i < epoch; i++) {
        auto loss = autograd::createValue(0, true);

        for (Data data : dataset) {
            auto y = autograd::createValue(data.y, false);
            auto yHat = x0 * data.x0 + x1 * data.x1 + b;
            loss = loss + autograd::pow(y - yHat, 2) / 2;
        }

        std::cout << "Epoch: " << i+1 << " Loss: " << loss->data << std::endl;
        loss->backward();

        x0->data -= learningRate * *x0->grad;
        x1->data -= learningRate * *x1->grad;
        b->data -= learningRate * *b->grad;

        loss->zeroGrad();
    }

    std::cout << "x0: " << x0->data << std::endl;
    std::cout << "x1: " << x1->data << std::endl;
    std::cout << "b: " << b->data << std::endl;

    return 0;
}
