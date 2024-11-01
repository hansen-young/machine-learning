#include <queue>
#include "autograd/autograd.h"
#include <iostream>

namespace autograd {
    Value::Value(double v) : data(v) {}
    Value::Value(double v, std::vector<Value*>& children, Operator* op) : data(v), children(children), op(op) {}
    Value::Value(const Value& other) { 
        data = other.data;
        
        if (other.grad != nullptr) { grad = new double(*(other.grad)); } 
        op = other.op; 
        children = other.children; 
    }


    void Value::printGraph() {
        std::queue<Value*> valueQueue;
        valueQueue.push(this);

        std::cout << "-------- CURRENT GRAPH --------" << std::endl;

        while (!valueQueue.empty()) {
            Value* current = valueQueue.front();
            valueQueue.pop();

            std::cout << "current: " << current->data << std::endl;
            if (current->op != nullptr) { std::cout << "     op: " << current->op->name << std::endl; }
            std::cout << "num_children: " << current->children.size() << std::endl;

            for(auto& child : current->children) {
                std::cout << "  child: " << child->data << std::endl;
            }

            for(auto& child : current->children) {
                valueQueue.push(child);
            }

            std::cout << std::endl;
        }

        std::cout << "-------------------------------\n";
    }
    
    void Value::backward() { 
        std::queue<Value*> valueQueue;
        valueQueue.push(this);

        while (!valueQueue.empty()) {
            Value* current = valueQueue.front();
            valueQueue.pop();

            // If this is a top-level value, then it is not derived from 
            // any other value and hence there is nothing to backpropagate.
            if (current->op == nullptr) { continue; }

            // If .backward() is already called, then raise an error.
            if (current->op->backwardCalled) {
                throw std::runtime_error(".backward() called more than once");
            }

            // If this is the leaf node, then the gradient is 1.
            if (current->grad == nullptr) { current->grad = new double(1); }

            // If the children of this value do not have a gradient, then initialize it to 0.
            for(auto& child : current->children) {
                if (child->grad == nullptr) { child->grad = new double(0); }
            }

            // Run the backward pass of the operator
            current->op->backward(current->grad); 

            // Mark the operator as backward called
            current->op->backwardCalled = true;

            // Add the children to the stack
            for(auto& child : current->children) {
                valueQueue.push(child);
            }
        }
    }

    Value Value::operator+(const Value& other) const {
        Value* this_ptr = const_cast<Value*>(this);
        Value* other_ptr = const_cast<Value*>(&other);

        std::vector<Value*> children = {this_ptr, other_ptr};
        Add* op = new Add(this_ptr, other_ptr);
        Value out(this->data + other.data, children, op);

        out.printGraph();

        return out;
    }

    Value Value::operator*(const Value& other) const {
        Value* this_ptr = const_cast<Value*>(this);
        Value* other_ptr = const_cast<Value*>(&other);

        std::vector<Value*> children = {this_ptr, other_ptr};
        Multiply* op = new Multiply(this_ptr, other_ptr);
        Value out(this->data * other.data, children, op);

        out.printGraph();

        return out;
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