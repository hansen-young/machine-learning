#include <iostream>
#include "autograd.h"


class Object : public std::enable_shared_from_this<Object> {
public:
    int x;
    std::vector<std::shared_ptr<Object>> children;

    // Constructors
    Object(int x) : x(x) {}
    Object(int x, std::vector<std::shared_ptr<Object>> children) : x(x), children(children) {}

    // Operator+ now accepts shared_ptrs
    std::shared_ptr<Object> operator+(std::shared_ptr<Object>& other) {
        // Create a new Object with this and other as children
        std::vector<std::shared_ptr<Object>> children = {
            shared_from_this(),
            other
        };
        return std::make_shared<Object>(this->x + other->x, children);
    }
};

#include <queue>
class ObjectWrapper {
public:
    std::shared_ptr<Object> obj;

    ObjectWrapper(int x) : obj(std::make_shared<Object>(x)) {}
    ObjectWrapper(std::shared_ptr<Object> obj) : obj(obj) {}

    // Operator+ now accepts ObjectWrapper
    ObjectWrapper operator+(ObjectWrapper& other) {
        return ObjectWrapper(obj->operator+(other.obj));
    }

    // Operator+ now accepts ObjectWrapper
    ObjectWrapper operator+(std::shared_ptr<Object>& other) {
        return ObjectWrapper(obj->operator+(other));
    }

    void printGraph() {
        std::queue<std::shared_ptr<Object>> objQueue;
        objQueue.push(obj);

        std::cout << "-------- CURRENT GRAPH --------" << std::endl;

        while (!objQueue.empty()) {
            std::shared_ptr<Object> current = objQueue.front();
            objQueue.pop();

            std::cout << "current: " << current->x << " | addr: " << current.get() << std::endl;
            std::cout << "num_children: " << current->children.size() << std::endl;

            for(std::shared_ptr<Object>& child : current->children) {
                std::cout << "  child: " << child->x << std::endl;
            }

            for(std::shared_ptr<Object>& child : current->children) {
                objQueue.push(child);
            }

            std::cout << std::endl;
        }

        std::cout << "-------------------------------\n";
    }
};

int main() {
    // auto a = std::make_shared<Object>(5);
    // auto b = std::make_shared<Object>(10);
    // auto c = std::make_shared<Object>(15);

    // auto d = *(*a + b) + c;
    // std::cout << d->x << std::endl;

    auto a = ObjectWrapper(5);
    auto b = ObjectWrapper(10);
    auto c = ObjectWrapper(15);
    auto d = a + b + c;

    // std::cout << d.obj->x << std::endl;
    // std::cout << d.obj->children[0]->x << std::endl;
    // std::cout << d.obj->children[0]->children.size() << std::endl;

    d.printGraph();

    return 0;
}

// int main() {
//     autograd::Value a(2);
//     autograd::Value b(3);
//     autograd::Value c(4);
//     autograd::Value d = a + b + c;

//     d.printGraph();

//     return 0;
// }
