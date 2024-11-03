#ifndef AUTOGRAD_VALUES_H
#define AUTOGRAD_VALUES_H

#include <vector>
#include <functional>
#include <memory>
#include <map>
#include "autograd/operators.h"

namespace autograd {
    class Operator;

    // Class definition
    class Value {
    private:
        static std::unordered_map< Value*, std::shared_ptr<Value> > instances;

        std::vector<Value*> children = std::vector<Value*>();
        Operator* op = nullptr;
        
    public:
        double data;
        double* grad = nullptr;

        Value(double v);
        Value(double v, std::vector<Value*>& children, Operator* op);
        ~Value();

        // Methods
        void backward();
        void printGraph();

        // Operators
        Value operator+(const Value& other) const;
        Value operator*(const Value& other) const;
    };
}

#endif // AUTOGRAD_VALUES_H