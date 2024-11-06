#ifndef AUTOGRAD_VALUES_H
#define AUTOGRAD_VALUES_H

#include <vector>
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
        bool requiresGrad = false;

        Value(double v);
        Value(double v, bool requiresGrad);
        Value(double v, std::vector<ValuePtr>& children, Operator* op, bool requiresGrad);
        ~Value();

        // Methods
        void backward();
        void zeroGrad();
        void printGraph();

        ValuePtr childAt(int index);
        int childrenSize();
    };

    // Functions
    ValuePtr createValue(double v, bool requiresGrad);
}

#endif // AUTOGRAD_VALUES_H