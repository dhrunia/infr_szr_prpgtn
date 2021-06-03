#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

int main(){
    // Eigen::Tensor<int, 1> a(4);
    // Eigen::Tensor<int, 1> b(4);
    // a.setValues({0, 1, 2, 3});
    // // Eigen::array<int, 2> offsets = {0, 0};
    // // Eigen::array<int, 2> extents = {1, 3};
    // // Eigen::Tensor<int, 1> slice = a.slice(offsets, extents);
    // const auto x = a.slice(Eigen::array<int, 1>({0}), Eigen::array<int, 1>({2}));
    // const auto z = a.slice(Eigen::array<int, 1>({2}), Eigen::array<int, 1>({2}));
    // std::cout << "a" << std::endl << a << std::endl;
    // std::cout << "x" << std::endl << x << std::endl;
    // std::cout << "z" << std::endl << z << std::endl;
    // b.slice(Eigen::array<int, 1>({0}), Eigen::array<int, 1>({2})) = z;
    // b.slice(Eigen::array<int, 1>({2}), Eigen::array<int, 1>({2})) = x;
    // std::cout << "b" << std::endl << b << std::endl;
    Eigen::Tensor<int, 1> a(4);
    a.setValues({0,1,2,3});
    // Eigen::TensorMap<Eigen::Tensor<int,2>> a_2d = a.reshape()
    auto b = a.reshape(Eigen::array<int, 2>({1,4})).broadcast(Eigen::array<int, 2>({5,1}));
    std::cout << a << std::endl << b << std::endl;
}