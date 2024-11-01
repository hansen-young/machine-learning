#ifndef AUTOGRAD_OPERATORS_H
#define AUTOGRAD_OPERATORS_H

#include <iostream>
#include <string>
#include "autograd/value.h"

namespace autograd {
    class Value;

    class Operator {
    public:
        const std::string name;
        bool backwardCalled = false;

        Operator(const std::string& name);
        virtual ~Operator() = default;
        virtual void backward(double* cum_grad) = 0;
    };

    class BinaryOperator : public Operator {
    protected:
        Value *left, *right;
    public:
        BinaryOperator(const std::string& name, Value* l, Value* r);
        virtual void backward(double* cum_grad) override = 0;
    };

    class Add : public BinaryOperator {
    public:
        Add(Value* l, Value* r);
        virtual void backward(double* cum_grad) override; 
    };

    class Multiply : public BinaryOperator {
    public:
        Multiply(Value* l, Value* r);
        virtual void backward(double* cum_grad) override;
    };
} // namespace autograd

#endif // AUTOGRAD_OPERATORS_H