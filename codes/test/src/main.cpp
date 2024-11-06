#include <iostream>
#include "autograd.h"


int main() {
    auto a = autograd::createValue(2, true);
    auto b = autograd::createValue(3, true);
    auto c = autograd::createValue(6, true);
    // auto d = a * b + a * c;
    // auto e = -a;
    // auto f = 5 + b;
    auto g = autograd::pow(a, 3);

    // d->backward();
    // d->printGraph();

    // e->backward();
    // e->printGraph();

    // f->backward();
    // f->printGraph();

    g->backward();
    g->printGraph();

    return 0;
}
