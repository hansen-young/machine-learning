#ifndef AUTOGRAD_OPERATORS_H
#define AUTOGRAD_OPERATORS_H

#include <iostream>
#include <string>
#include "autograd/value.h"

namespace autograd {
    // Forward declaration
    class Value;

    // Aliases
    using ValuePtr = std::shared_ptr<Value>;

    // Class definition
    class Operator {
    public:
        const std::string name;

        Operator(const std::string& name) : name(name) {};
        virtual ~Operator() = default;
        virtual void backward(double* cum_grad, std::vector<ValuePtr>& children) = 0;
    };

    class _Add : public Operator {
    public:
        _Add() : Operator("Add") {};
        virtual void backward(double* cum_grad, std::vector<ValuePtr>& children) override;
    };

    class _Multiply : public Operator {
    public:
        _Multiply() : Operator("Multiply") {};
        virtual void backward(double* cum_grad, std::vector<ValuePtr>& children) override;
    };

    // Declare the static instances as extern
    extern _Add Add;
    extern _Multiply Multiply;
} // namespace autograd

#endif // AUTOGRAD_OPERATORS_H