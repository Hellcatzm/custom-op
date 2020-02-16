#include <iostream>

int main(){
    int ar[5]={0, 1, 1, 0, 1};
    
    for (auto i : ar)
        if (i)
            std::cout << i << std::endl;
    return 0;
};
