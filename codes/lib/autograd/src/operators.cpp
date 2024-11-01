#include "autograd/operators.h"
#include <iostream>

namespace autograd {
    Operator::Operator(const std::string& name) : name(name) {}
    BinaryOperator::BinaryOperator(const std::string& name, Value* l, Value* r) : Operator(name), left(l), right(r) {}

    Add::Add(Value* l, Value* r) : BinaryOperator("Add", l, r) { std::cout << "Add(" << l->data << ", " << r->data << ") created." << std::endl; }
    void Add::backward(double* cum_grad) { 
        *(this->left->grad) += 1 * *cum_grad;
        *(this->right->grad) += 1 * *cum_grad;
    }

    Multiply::Multiply(Value* l, Value* r) : BinaryOperator("Multiply", l, r) { std::cout << "Multiply(" << l->data << ", " << r->data << ") created." << std::endl; }
    void Multiply::backward(double* cum_grad) { 
        *(this->left->grad) += this->right->data * *cum_grad;
        *(this->right->grad) += this->left->data * *cum_grad;
    }
} // namespace autograd