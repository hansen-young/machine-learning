#include <iostream>
#include <queue>
#include "autograd/autograd.h"


namespace autograd {
    Value::Value(double v) : data(v) {}
    Value::Value(double v, bool requiresGrad) : data(v), requiresGrad(requiresGrad) {}
    Value::Value(double v, std::vector<ValuePtr>& children, Operator* op, bool requiresGrad) : data(v), children(children), op(op), requiresGrad(requiresGrad) {}
    // Value::~Value() { std::cout << "Deconstructor of Value(x=" << this->data << ")" << std::endl; }
    Value::~Value() {}

    void Value::printGraph() {
        std::queue<ValuePtr> valueQueue;
        valueQueue.push(shared_from_this());

        std::cout << "-------- CURRENT GRAPH --------" << std::endl;

        while (!valueQueue.empty()) {
            ValuePtr current = valueQueue.front();
            valueQueue.pop();

            std::cout << current->data << " (" << current << ") | grad: ";
            if (current->grad != nullptr) { std::cout << *current->grad << std::endl; }
            else { std::cout << "null" << std::endl; }

            if (current->op != nullptr) { std::cout << "op: " << current->op->name << std::endl; }
            std::cout << "num_children: " << current->children.size() << std::endl;

            for(ValuePtr child : current->children) {
                std::cout << "  child: " << child->data << " (" << child << ")" << std::endl;
                valueQueue.push(child);
            }

            std::cout << std::endl;
        }
        std::cout << "-------------------------------\n";
    }

    void Value::backward() {
        std::queue<ValuePtr> valueQueue;
        valueQueue.push(shared_from_this());

        while (!valueQueue.empty()) {
            ValuePtr current = valueQueue.front();
            valueQueue.pop();

            // If this node does not require gradient, then skip it.
            if (!current->requiresGrad) { continue; }

            // If this is a top-level value, then it is not derived from
            // any other value and hence there is nothing to backpropagate.
            if (current->op == nullptr) { continue; }

            // If .backward() is already called, then raise an error.
            if (current->backwardCalled) {
                throw std::runtime_error(".backward() called more than once");
            }

            // If this is the leaf node, then the gradient is 1.
            if (current->grad == nullptr) { current->grad = new double(1); }

            // If the children of this value do not have a gradient, then initialize it to 0.
            for(ValuePtr child : current->children) {
                if (child->requiresGrad && child->grad == nullptr) { child->grad = new double(0); }
            }

            // Run the backward pass of the operator
            current->op->backward(current);
            current->backwardCalled = true;

            // Add the children to the stack
            for(ValuePtr child : current->children) {
                valueQueue.push(child);
            }
        }
    }

    void Value::zeroGrad() {
        std::queue<ValuePtr> valueQueue;
        valueQueue.push(shared_from_this());

        while (!valueQueue.empty()) {
            ValuePtr current = valueQueue.front();
            valueQueue.pop();

            if (current->grad != nullptr) {
                current->grad = nullptr;
            }

            for(ValuePtr child : current->children) {
                if (child->requiresGrad) {
                    valueQueue.push(child);
                }
            }
        }
    }

    ValuePtr Value::childAt(int index) { return this->children[index]; }

    int Value::childrenSize() { return this->children.size(); }

    ValuePtr createValue(double v, bool requiresGrad) {
        return std::make_shared<Value>(v, requiresGrad);
    }
} // namespace autograd


/*
d = a * b + a * c

b = 3 -- * -- a = 2  -- * -- c = 6
         |              |
         |              |
      x0 = 6 -- + -- x1 = 12
                |
              d = 18

grad(a) = dd/da = dx0/da * dd/dx0 + dx1/da * dd/dx1
                = b * grad(x0) + c * grad(x1)
grad(x0) = dd/dx0 = dd/dx0 * dd/dd = 1 * 1 = 1
grad(x1) = dd/dx1 = dd/dx1 * dd/dd = 1 * 1 = 1
grad(a) = 3 * 1 + 6 * 1 = 9

----

d.children = [x0, x1]
d.operator = Add(x0, x1)

x1.children = [a, c]
x1.operator = Multiply(a, c)

x0.children = [a, b]
x0.operator = Multiply(a, b)

stack = [d]

// if d.grad == nullptr then d.grad = 1
Add.backward(d.grad) will compute
    x0.grad += 1 * d.grad
    x1.grad += 1 * d.grad

stack = [x0, x1]
*/