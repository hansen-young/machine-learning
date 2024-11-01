#ifndef AUTOGRAD_VALUES_H
#define AUTOGRAD_VALUES_H

#include <vector>
#include <functional>
#include "autograd/operators.h"

namespace autograd {
    class Operator;

    class Value {
    private:
        
        
    public:
        double data;
        double* grad = nullptr;
        
        std::vector<Value*> children = std::vector<Value*>();
        Operator* op = nullptr;

        Value(double v);
        Value(double v, std::vector<Value*>& children, Operator* op);
        Value(const Value& other);

        // Methods
        void backward();
        void printGraph();

        // Operators
        Value operator+(const Value& other) const;
        Value operator*(const Value& other) const;
    };
}

#endif // AUTOGRAD_VALUES_H