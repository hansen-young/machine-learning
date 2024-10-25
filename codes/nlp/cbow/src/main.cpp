#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>
#include "utils.cpp"



/**
 * @class ContinuousBagOfWords
 * @brief A class representing the CBOW model used in natural language processing.
 * 
 * The ContinuousBagOfWords class is used to implement the CBOW model, which is a type of neural network
 * used for word embedding in natural language processing tasks.
 * 
 * @param feature_size The size of the feature vector
 * @param window_size The size of the window used to generate the context words
 * 
 */
class ContinuousBagOfWords {
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

        std::vector<FloatVector> computeVectorAverage(std::vector<FloatVector> vectors, int window_size) {
            std::vector<FloatVector> result;

            // Create a mask [1, ..., 1, 0, 1, ..., 1]
            std::vector<float> mask(window_size, 1.0);
            mask.push_back(0.0);
            mask.insert(mask.end(), window_size, 1.0);
            float mask_sum = std::accumulate(mask.begin(), mask.end(), 0.0);

            // Convolve the mask with the vectors
            for (int i = 0; i < vectors.size(); ++i) {
                FloatVector vec;
                for (int j = 0; j < vectors[i].size(); ++j) {
                    float sum = 0.0;

                    for (int k = 0; k < mask.size(); ++k) {
                        size_t jj = j - window_size + k;
                        if (jj < 0 || jj >= vectors[i].size()) {continue;}
                        sum += vectors[i][jj] * mask[k];
                    }
                    
                    vec.push_back(sum / mask_sum);
                }
                result.push_back(vec);
            }

            return result;
        }

    public:
        int feature_size;
        int window_size;
        
        ContinuousBagOfWords(int feature_size, int window_size) {
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
                
                // Compute V_bar
                std::vector<FloatVector> vBar = computeVectorAverage(v, window_size);

                // Compute U * V_bar
                std::vector<FloatVector> u_dot_vbar(u.size(), FloatVector());
                for (int iu = 0; iu < u.size(); ++iu) {
                    for (int iv = 0; iv < vBar.size(); ++iv) {
                        u_dot_vbar[iu].push_back(u[iu].dot(vBar[iv]));
                    }
                }

                // Compute P(wc | Wo)
                std::vector<FloatVector> p;
                for (int iv = 0; iv < vBar.size(); ++iv) {
                    p.push_back(FloatVector());
                    float denom = 0.0;

                    for (int iu = 0; iu < u.size(); ++iu) {
                        denom += exp(u_dot_vbar[iu][iv]);
                    }

                    for (int iu = 0; iu < u.size(); ++iu) {
                        p[iv].push_back(exp(u_dot_vbar[iu][iv]) / denom);
                    }
                }

                // Initialize the loss
                float loss = 0.0;

                // Iterate over the training data
                for (std::vector<unsigned int>& indexes : trainData) {

                    // Iterate over the center words
                    for (int t = 0; t < indexes.size(); ++t) {

                        // Compute the loss
                        unsigned int c = indexes[t];
                        loss += -log(p[c][c]);

                        // Compute the gradients for v_indexes[x] (t - m <= x <= t + m)
                        for (int w = -window_size; w <= window_size; ++w) {
                            if (w == 0 || t + w < 0 || t + w >= indexes.size()) {continue;}
                            unsigned int o = indexes[t + w];
                            FloatVector curr_loss = FloatVector(feature_size, 0.0) - u[c];

                            for (int j = 0; j < u.size(); ++j) {
                                curr_loss = curr_loss + u[j] * p[c][j]; 
                            }

                            grad_v[o] = grad_v[o] + curr_loss / (2 * window_size);
                        }

                        // Compute the gradients for u_c
                        grad_u[c] = grad_u[c] + vBar[c] * (p[c][c] - 1);

                        // Compute the gradients for u_k (k != c)
                        for (int k = 0; k < u.size(); ++k) {
                            if (k == c) {continue;}
                            grad_u[k] = grad_u[k] + vBar[c] * p[c][k];
                        }
                    }
                }

                // Update the vectors u and v
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
    ContinuousBagOfWords model(20, 2);
    model.fit(X, 250, 0.005);
    model.save("model");

    return 0;
}