#include "autograd/operators.h"
#include <iostream>
#include <memory>
#include <cmath>

namespace autograd {
    // Define the static instances
    _UnaryMinus UnaryMinus;
    _Add Add;
    _Subtract Subtract;
    _Multiply Multiply;
    _Divide Divide;
    _Pow Pow;
    _Sqrt Sqrt;

    // Helper functions
    void throwIfChildrenNotEqual(Operator* o, int childrenSize, int expected) {
        if (childrenSize != expected) {
            throw std::invalid_argument(
                "Operator " + o->name + " must have exactly " + std::to_string(expected) + " children. Got " + std::to_string(childrenSize) + "."
            );
        }
    }

    // Backward functions
    void _UnaryMinus::backward(ValuePtr node) {
        // y = -a -> dy/da = -1
        throwIfChildrenNotEqual(this, node->childrenSize(), 1);
        if (node->childAt(0)->requiresGrad){ *(node->childAt(0)->grad) += -1 * *(node->grad); }
    }

    void _Add::backward(ValuePtr node) {
        // y = a + b -> dy/da = 1
        //           -> dy/db = 1
        throwIfChildrenNotEqual(this, node->childrenSize(), 2);
        if (node->childAt(0)->requiresGrad){ *(node->childAt(0)->grad) += 1 * *(node->grad); }
        if (node->childAt(1)->requiresGrad){ *(node->childAt(1)->grad) += 1 * *(node->grad); }
    }

    void _Subtract::backward(ValuePtr node) {
        // y = a - b -> dy/da = 1
        //           -> dy/db = -1
        throwIfChildrenNotEqual(this, node->childrenSize(), 2);
        if (node->childAt(0)->requiresGrad){ *(node->childAt(0)->grad) += 1 * *(node->grad); }
        if (node->childAt(1)->requiresGrad){ *(node->childAt(1)->grad) += -1 * *(node->grad); }
    }

    void _Multiply::backward(ValuePtr node) {
        // y = a * b -> dy/da = b
        //           -> dy/db = a
        throwIfChildrenNotEqual(this, node->childrenSize(), 2);
        if (node->childAt(0)->requiresGrad){ *(node->childAt(0)->grad) += node->childAt(1)->data * *(node->grad); }
        if (node->childAt(1)->requiresGrad){ *(node->childAt(1)->grad) += node->childAt(0)->data * *(node->grad); }
    }

    void _Divide::backward(ValuePtr node) {
        // y = a / b -> dy/da = 1 / b
        //           -> dy/db = -a / (b^2)
        throwIfChildrenNotEqual(this, node->childrenSize(), 2);
        if (node->childAt(0)->requiresGrad){ *(node->childAt(0)->grad) += 1 / node->childAt(1)->data * *(node->grad); }
        if (node->childAt(1)->requiresGrad){ *(node->childAt(1)->grad) += -node->childAt(0)->data / (node->childAt(1)->data * node->childAt(1)->data) * *(node->grad); }
    }

    void _Pow::backward(ValuePtr node) {
        // y = a ^ b -> dy/da = b * (a ^ (b - 1)) = b * y / a
        //           -> dy/db = (a ^ b) * log(a) = y * log(a)
        throwIfChildrenNotEqual(this, node->childrenSize(), 2);
        if (node->childAt(0)->requiresGrad){
            *(node->childAt(0)->grad) += node->childAt(1)->data * node->data / node->childAt(0)->data * *(node->grad);
        }
        if (node->childAt(1)->requiresGrad){
            *(node->childAt(1)->grad) += node->data * std::log(node->childAt(0)->data) * *(node->grad);
        }
    }

    void _Sqrt::backward(ValuePtr node) {
        // y = sqrt(a) -> dy/da = 1 / (2 * sqrt(a)) = 1 / (2 * y)
        throwIfChildrenNotEqual(this, node->childrenSize(), 1);
        if (node->childAt(0)->requiresGrad){
            *(node->childAt(0)->grad) += 1 / (2 * node->data) * *(node->grad);
        }
    }

    // Functions
    ValuePtr operator-(ValuePtr a) {
        std::vector<ValuePtr> children = {a};
        return std::make_shared<Value>(-a->data, children, &UnaryMinus, a->requiresGrad);
    }

    ValuePtr operator+(ValuePtr a, ValuePtr b) {
        std::vector<ValuePtr> children = {a, b};
        return std::make_shared<Value>(a->data + b->data, children, &Add, a->requiresGrad | b->requiresGrad);
    }

    ValuePtr operator-(ValuePtr a, ValuePtr b) {
        std::vector<ValuePtr> children = {a, b};
        return std::make_shared<Value>(a->data - b->data, children, &Subtract, a->requiresGrad | b->requiresGrad);
    }

    ValuePtr operator*(ValuePtr a, ValuePtr b) {
        std::vector<ValuePtr> children = {a, b};
        return std::make_shared<Value>(a->data * b->data, children, &Multiply, a->requiresGrad | b->requiresGrad);
    }

    ValuePtr operator/(ValuePtr a, ValuePtr b) {
        std::vector<ValuePtr> children = {a, b};
        return std::make_shared<Value>(a->data / b->data, children, &Divide, a->requiresGrad | b->requiresGrad);
    }

    ValuePtr pow(ValuePtr a, ValuePtr b) {
        std::vector<ValuePtr> children = {a, b};
        return std::make_shared<Value>(std::pow(a->data, b->data), children, &Pow, a->requiresGrad | b->requiresGrad);
    }

    ValuePtr sqrt(ValuePtr a) {
        std::vector<ValuePtr> children = {a};
        return std::make_shared<Value>(std::sqrt(a->data), children, &Sqrt, a->requiresGrad);
    }

    ValuePtr operator+(ValuePtr a, double scalar) {
        std::vector<ValuePtr> children = {a, std::make_shared<Value>(scalar)};
        return std::make_shared<Value>(a->data + scalar, children, &Add, a->requiresGrad);
    }

    ValuePtr operator+(double scalar, ValuePtr a) { return a + scalar; }

    ValuePtr operator-(ValuePtr a, double scalar) {
        std::vector<ValuePtr> children = {a, std::make_shared<Value>(scalar)};
        return std::make_shared<Value>(a->data - scalar, children, &Subtract, a->requiresGrad);
    }

    ValuePtr operator-(double scalar, ValuePtr a) {
        std::vector<ValuePtr> children = {std::make_shared<Value>(scalar), a};
        return std::make_shared<Value>(scalar - a->data, children, &Subtract, a->requiresGrad);
    }

    ValuePtr operator*(ValuePtr a, double scalar) {
        std::vector<ValuePtr> children = {a, std::make_shared<Value>(scalar)};
        return std::make_shared<Value>(a->data * scalar, children, &Multiply, a->requiresGrad);
    }

    ValuePtr operator*(double scalar, ValuePtr a) { return a * scalar; }

    ValuePtr operator/(ValuePtr a, double scalar) {
        std::vector<ValuePtr> children = {a, std::make_shared<Value>(scalar)};
        return std::make_shared<Value>(a->data / scalar, children, &Divide, a->requiresGrad);
    }

    ValuePtr operator/(double scalar, ValuePtr a) {
        std::vector<ValuePtr> children = {std::make_shared<Value>(scalar), a};
        return std::make_shared<Value>(scalar / a->data, children, &Divide, a->requiresGrad);
    }

    ValuePtr pow(ValuePtr a, double scalar) {
        std::vector<ValuePtr> children = {a, std::make_shared<Value>(scalar)};
        return std::make_shared<Value>(std::pow(a->data, scalar), children, &Pow, a->requiresGrad);
    }

    ValuePtr pow(double scalar, ValuePtr a) {
        std::vector<ValuePtr> children = {std::make_shared<Value>(scalar), a};
        return std::make_shared<Value>(std::pow(scalar, a->data), children, &Pow, a->requiresGrad);
    }

    ValuePtr sqrt(double scalar) {
        return std::make_shared<Value>(std::sqrt(scalar));
    }
} // namespace autograd