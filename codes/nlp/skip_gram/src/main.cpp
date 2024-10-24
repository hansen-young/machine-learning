#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include "utils.cpp"


/**
 * @class SkipGram
 * @brief A class representing the SkipGram model used in natural language processing.
 * 
 * The SkipGram class is used to implement the SkipGram model, which is a type of neural network
 * used for word embedding in natural language processing tasks.
 * 
 * @param feature_size The size of the feature vector
 * @param window_size The size of the window used to generate the context words
 * 
 */
class SkipGram {
    private:
        std::vector<FloatVector> u;
        std::vector<FloatVector> v;
        std::unordered_map<std::string, unsigned int> dictionary;

        FloatVector initializeRandomVector(int size) {
            FloatVector vec;
            for (int i = 0; i < size; i++) {
                vec.push_back((float) rand() / RAND_MAX);
            }
            return vec;
        }

        unsigned int createWord(std::string word) {
            if (dictionary.find(word) == dictionary.end()) {
                dictionary[word] = dictionary.size();
                u.push_back(initializeRandomVector(feature_size));
                v.push_back(initializeRandomVector(feature_size));
            }
            return dictionary[word];
        }

        std::vector<unsigned int> sentenceToIndexes(std::string sentence) {
            std::vector<std::string> tokens = splitSentenceToWords(sentence);
            std::vector<unsigned int> indexes;

            for(std::string token : tokens) {
                unsigned int index = createWord(token);
                indexes.push_back(dictionary[token]);
            }

            return indexes;
        }

    public:
        int feature_size;
        int window_size;
        
        SkipGram(int feature_size, int window_size) {
            this->feature_size = feature_size;
            this->window_size = window_size;
        }

        void fit(std::vector<std::string> X, int epochs = 10, float lr = 0.01) {
            // Create the dictionary
            std::vector< std::vector<unsigned int> > trainData;
            for (std::string sentence : X) {trainData.push_back(sentenceToIndexes(sentence));}

            // Initialize gradients vectors
            std::vector<FloatVector> grad_u(u.size(), FloatVector(feature_size, 0.0));
            std::vector<FloatVector> grad_v(v.size(), FloatVector(feature_size, 0.0));

            // Train the model
            for (int epoch = 0; epoch < epochs; ++epoch) {
                // Compute U * V
                std::vector<FloatVector> u_dot_v(u.size(), FloatVector());
                for (int iu = 0; iu < u.size(); ++iu) {
                    for (int iv = 0; iv < v.size(); ++iv) {
                        u_dot_v[iu].push_back(u[iu].dot(v[iv]));
                    }
                }

                // Compute P(wo | wc)
                std::vector<FloatVector> p;
                for (int ic = 0; ic < v.size(); ++ic) {
                    p.push_back(FloatVector());
                    float denom = 0.0;

                    for (FloatVector ui_dot_v : u_dot_v) {
                        denom += exp(ui_dot_v[ic]);
                    }

                    for (int io = 0; io < u.size(); ++io) {
                        p[ic].push_back(exp(u_dot_v[io][ic]) / denom);
                    }
                }

                // Initialize the loss
                float loss = 0.0;

                // Iterate over the training data
                for (std::vector<unsigned int>& indexes : trainData) {

                    // Iterate over the center words
                    for (int t = 0; t < indexes.size(); ++t) {

                        // Iterate over the context words
                        for (int j = -window_size; j <= window_size; ++j) {
                            if (j == 0 || t + j < 0 || t + j >= indexes.size()) {continue;}

                            unsigned int c = indexes[t];
                            unsigned int o = indexes[t + j];

                            // Compute the loss
                            loss -= log(p[c][o]);

                            // Compute the gradients for v_c
                            grad_v[c] = grad_v[c] - u[o];
                            for (int z = 0; z < grad_u.size(); ++z) {
                                grad_v[c] = grad_v[c] + u[z] * p[c][z];
                            }

                            // Compute the gradients for u_o
                            grad_u[o] = grad_u[0] + v[c] * (p[c][o] - 1);

                            // Compute the gradients for u_z (z != o)
                            for (int z = 0; z < grad_u.size(); ++z) {
                                if (z == o) {continue;}
                                grad_u[z] = grad_u[z] + v[c] * p[c][z];
                            }
                        }
                    }
                }

                // Update vector u and v
                for (int i = 0; i < u.size(); ++i) {
                    u[i] = u[i] - grad_u[i] * lr;
                    v[i] = v[i] - grad_v[i] * lr;
                }

                // Clear the gradients
                for (int i = 0; i < u.size(); ++i) {
                    grad_u[i] = FloatVector(feature_size, 0.0);
                    grad_v[i] = FloatVector(feature_size, 0.0);
                }

                std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << loss << std::endl;
            }
        }

        void save(std::string directory) {
            // Create directory if it does not exist 
            std::filesystem::create_directory(directory);

            // Save the dictionary
            std::ofstream fp(directory + "/dictionary.txt");

            for (auto const& [word, index] : dictionary) {
                fp << word << " " << index << std::endl;
            }

            // Save vector u
            fp = std::ofstream(directory + "/u.txt");
            for (FloatVector& vec : u) {
                for (float value : vec) {
                    fp << value << " ";
                }
                fp << std::endl;
            }

            // Save vector v
            fp = std::ofstream(directory + "/v.txt");
            for (FloatVector& vec : v) {
                for (float value : vec) {
                    fp << value << " ";
                }
                fp << std::endl;
            }

            fp.close();
        }
};


int main() {
    // Read from file dataset.txt
    std::vector<std::string> X;
    std::string line;
    std::ifstream fp("dataset/train.txt");
    
    while (getline(fp, line)) {
        X.push_back(line);
    }
    fp.close();

    // Create the SkipGram model
    SkipGram model(20, 2);
    model.fit(X, 250, 0.005);
    model.save("model");

    return 0;
}