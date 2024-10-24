#include <vector>
#include <string>
#include <sstream>


class FloatVector : public std::vector<float> {
    public:
        using std::vector<float>::vector;

        FloatVector operator+(const FloatVector& other) {
            FloatVector result;
            for (int i = 0; i < this->size(); i++) {
                result.push_back(this->at(i) + other.at(i));
            }
            return result;
        }

        FloatVector operator-(const FloatVector& other) {
            FloatVector result;
            for (int i = 0; i < this->size(); i++) {
                result.push_back(this->at(i) - other.at(i));
            }
            return result;
        }

        FloatVector operator*(const FloatVector& other) {
            FloatVector result;
            for (int i = 0; i < this->size(); i++) {
                result.push_back(this->at(i) * other.at(i));
            }
            return result;
        }

        FloatVector operator*(const float& scalar) {
            FloatVector result;
            for (int i = 0; i < this->size(); i++) {
                result.push_back(this->at(i) * scalar);
            }
            return result;
        }

        float dot(const FloatVector& other) {
            float result = 0;
            for (int i = 0; i < this->size(); i++) {
                result += this->at(i) * other.at(i);
            }
            return result;
        }

        float sum() {
            float result = 0;
            for (float value : *this) {
                result += value;
            }
            return result;
        }
};


// Function to split a sentence into word tokens
std::vector<std::string> splitSentenceToWords(const std::string& sentence) {
    std::vector<std::string> words;
    std::istringstream stream(sentence);
    std::string word;
    
    while (stream >> word) {
        words.push_back(word);
    }
    
    return words;
}