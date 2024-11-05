#ifndef AUTOGRAD_VALUES_H
#define AUTOGRAD_VALUES_H

#include <vector>
#include <functional>
#include <memory>
#include <map>
#include "autograd/operators.h"

namespace autograd {
    // Forward declaration
    class Operator;
    class Value;

    // Aliases
    using ValuePtr = std::shared_ptr<Value>;

    // Class definition
    class Value : public std::enable_shared_from_this<Value> {
    private:
        std::vector<ValuePtr> children = std::vector<ValuePtr>();
        Operator* op = nullptr;
        bool backwardCalled = false;

    public:
        double data;
        double* grad = nullptr;

        Value(double v);
        Value(double v, std::vector<ValuePtr>& children, Operator* op);
        ~Value();

        // Methods
        void backward();
        void printGraph();
    };

    // Functions
    ValuePtr createValue(double v);
}

#endif // AUTOGRAD_VALUES_H