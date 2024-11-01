#include <iostream>
#include "autograd.h"



int main() {
    autograd::Value a(2);
    autograd::Value b(3);
    autograd::Value c(6);

    // autograd::Value x0 = a * b;
    // autograd::Value x1 = a * c;
    // autograd::Value d = x0 + x1;

    autograd::Value d = a * b + a * c;
    // autograd::Value d = a * b * c;

    std::cout << "FROM MAIN" << std::endl;
    d.printGraph();

    // std::cout << "d.children: " << d.children.size() << std::endl;
    // std::cout << "a * b children: " << d.children[0]->children.size() << std::endl;
    // std::cout << "a * c children: " << d.children[1]->children.size() << std::endl;

    // std::cout << "a * b operator: " << std::endl; d.children[0]->op->printChildren();
    // std::cout << "a * c operator: " << std::endl; d.children[1]->op->printChildren();



    // std::cout << "d = " << d.data << std::endl;

    // d.backward();

    // std::cout << "grad(a) = " << *(a.grad) << std::endl;
    // std::cout << "grad(b) = " << *(b.grad) << std::endl;
    // std::cout << "grad(c) = " << *(c.grad) << std::endl;
    // std::cout << "grad(d) = " << *(d.grad) << std::endl;

    return 0;
}
