#include <iostream>
#include "autograd.h"


int main() {
    auto a = autograd::createValue(2);
    auto b = autograd::createValue(3);
    auto c = autograd::createValue(6);
    auto d = a * b + a * c;

    d->backward();
    d->printGraph();

    return 0;
}
