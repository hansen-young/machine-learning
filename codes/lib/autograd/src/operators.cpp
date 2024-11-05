#include "autograd/operators.h"
#include <iostream>

namespace autograd {
    // Define the static instances
    _Add Add;
    _Multiply Multiply;

    // Helper functions
    void throwIfChildrenNotEqual(Operator* o, std::vector<ValuePtr>& children, int expected) {
        if (children.size() != expected) {
            throw std::invalid_argument(
                "Operator " + o->name + " must have exactly " + std::to_string(expected) + " children. Got " + std::to_string(children.size()) + "."
            );
        }
    }

    // Backward functions
    void _Add::backward(double* cum_grad, std::vector<ValuePtr>& children) {
        throwIfChildrenNotEqual(this, children, 2);
        *(children[0]->grad) += 1 * *cum_grad;
        *(children[1]->grad) += 1 * *cum_grad;
    }

    void _Multiply::backward(double* cum_grad, std::vector<ValuePtr>& children) {
        throwIfChildrenNotEqual(this, children, 2);
        *(children[0]->grad) += children[1]->data * *cum_grad;
        *(children[1]->grad) += children[0]->data * *cum_grad;
    }
} // namespace autograd