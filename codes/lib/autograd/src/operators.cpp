#include "autograd/operators.h"
#include <iostream>

namespace autograd {
    Operator::Operator(const std::string& name) : name(name) {}

    void throwIfChildrenNotEqual(Operator* o, std::vector<Value*>& children, int expected) {
        if (children.size() != expected) {
            throw std::invalid_argument(
                "Operator " + o->name + " must have exactly " + std::to_string(expected) + " children. Got " + std::to_string(children.size()) + "."
            );
        }
    }

    // Define the static instances
    _Add Add;
    _Multiply Multiply;

    // Implement the backward method for each operator
    void _Add::backward(double* cum_grad, std::vector<Value*>& children) {
        std::cout << "Size: " << children.size() << std::endl;
        throwIfChildrenNotEqual(this, children, 2);
        std::cout << "Child[0]: " << children[0]->data << " | Grad: " << *children[0]->grad << std::endl;
        std::cout << "Child[1]: " << children[1]->data << std::endl;
        *(children[0]->grad) += 1 * *cum_grad;
        *(children[1]->grad) += 1 * *cum_grad;
    }

    void _Multiply::backward(double* cum_grad, std::vector<Value*>& children) { 
        throwIfChildrenNotEqual(this, children, 2);
        *(children[0]->grad) += children[1]->data * *cum_grad;
        *(children[1]->grad) += children[0]->data * *cum_grad;
    }
} // namespace autograd