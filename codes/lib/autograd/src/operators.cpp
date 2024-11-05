#include "autograd/operators.h"
#include <iostream>

namespace autograd {
    // Define the static instances
    _UnaryMinus UnaryMinus;
    _Add Add;
    _Subtract Subtract;
    _Multiply Multiply;
    _Divide Divide;

    // Helper functions
    void throwIfChildrenNotEqual(Operator* o, std::vector<ValuePtr>& children, int expected) {
        if (children.size() != expected) {
            throw std::invalid_argument(
                "Operator " + o->name + " must have exactly " + std::to_string(expected) + " children. Got " + std::to_string(children.size()) + "."
            );
        }
    }

    // Backward functions
    void _UnaryMinus::backward(double* cum_grad, std::vector<ValuePtr>& children) {
        throwIfChildrenNotEqual(this, children, 1);
        *(children[0]->grad) += -1 * *cum_grad;
    }

    void _Add::backward(double* cum_grad, std::vector<ValuePtr>& children) {
        throwIfChildrenNotEqual(this, children, 2);
        *(children[0]->grad) += 1 * *cum_grad;
        *(children[1]->grad) += 1 * *cum_grad;
    }

    void _Subtract::backward(double* cum_grad, std::vector<ValuePtr>& children) {
        throwIfChildrenNotEqual(this, children, 2);
        *(children[0]->grad) += 1 * *cum_grad;
        *(children[1]->grad) += -1 * *cum_grad;
    }

    void _Multiply::backward(double* cum_grad, std::vector<ValuePtr>& children) {
        throwIfChildrenNotEqual(this, children, 2);
        *(children[0]->grad) += children[1]->data * *cum_grad;
        *(children[1]->grad) += children[0]->data * *cum_grad;
    }

    void _Divide::backward(double* cum_grad, std::vector<ValuePtr>& children) {
        throwIfChildrenNotEqual(this, children, 2);
        *(children[0]->grad) += 1 / children[1]->data * *cum_grad;
        *(children[1]->grad) += -children[0]->data / (children[1]->data * children[1]->data) * *cum_grad;
    }

    // Functions
    ValuePtr operator-(ValuePtr a) {
        std::vector<ValuePtr> children = {a};
        return std::make_shared<Value>(-a->data, children, &UnaryMinus);
    }

    ValuePtr operator+(ValuePtr a, ValuePtr b) {
        std::vector<ValuePtr> children = {a, b};
        return std::make_shared<Value>(a->data + b->data, children, &Add);
    }

    ValuePtr operator-(ValuePtr a, ValuePtr b) {
        std::vector<ValuePtr> children = {a, b};
        return std::make_shared<Value>(a->data - b->data, children, &Subtract);
    }

    ValuePtr operator*(ValuePtr a, ValuePtr b) {
        std::vector<ValuePtr> children = {a, b};
        return std::make_shared<Value>(a->data * b->data, children, &Multiply);
    }

    ValuePtr operator/(ValuePtr a, ValuePtr b) {
        std::vector<ValuePtr> children = {a, b};
        return std::make_shared<Value>(a->data / b->data, children, &Divide);
    }
} // namespace autograd