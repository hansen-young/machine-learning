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
        virtual void backward(ValuePtr node) = 0;
    };

    class _UnaryMinus : public Operator {
    public:
        _UnaryMinus() : Operator("UnaryMinus") {};
        virtual void backward(ValuePtr node) override;
    };

    class _Add : public Operator {
    public:
        _Add() : Operator("Add") {};
        virtual void backward(ValuePtr node) override;
    };

    class _Subtract : public Operator {
    public:
        _Subtract() : Operator("Subtract") {};
        virtual void backward(ValuePtr node) override;
    };

    class _Multiply : public Operator {
    public:
        _Multiply() : Operator("Multiply") {};
        virtual void backward(ValuePtr node) override;
    };

    class _Divide : public Operator {
    public:
        _Divide() : Operator("Divide") {};
        virtual void backward(ValuePtr node) override;
    };

    class _Pow : public Operator {
    public:
        _Pow() : Operator("Pow") {};
        virtual void backward(ValuePtr node) override;
    };

    // Declare the static instances as extern
    extern _UnaryMinus UnaryMinus;
    extern _Add Add;
    extern _Subtract Subtract;
    extern _Multiply Multiply;
    extern _Divide Divide;
    extern _Pow Pow;

    // Functions
    ValuePtr operator-(ValuePtr a);  // Unary minus (negation)
    ValuePtr operator+(ValuePtr a, ValuePtr b);
    ValuePtr operator-(ValuePtr a, ValuePtr b);
    ValuePtr operator*(ValuePtr a, ValuePtr b);
    ValuePtr operator/(ValuePtr a, ValuePtr b);
    ValuePtr pow(ValuePtr a, ValuePtr b);

    ValuePtr operator+(ValuePtr a, double scalar);
    ValuePtr operator+(double scalar, ValuePtr a);
    ValuePtr operator-(ValuePtr a, double scalar);
    ValuePtr operator-(double scalar, ValuePtr a);
    ValuePtr operator*(ValuePtr a, double scalar);
    ValuePtr operator*(double scalar, ValuePtr a);
    ValuePtr operator/(ValuePtr a, double scalar);
    ValuePtr operator/(double scalar, ValuePtr a);
    ValuePtr pow(ValuePtr a, double scalar);
    ValuePtr pow(double scalar, ValuePtr a);
} // namespace autograd

#endif // AUTOGRAD_OPERATORS_H