#include "nntile/constants.hh"
#include "testing.hh"

using namespace nntile;

int main(int argc, char **argv)
{
    for(int i = -10; i < 10; ++i)
    {
        auto value = static_cast<enum TransOp::Value>(i);
        switch(value)
        {
            case TransOp::NoTrans:
                std::cout << "NT\n";
            case TransOp::Trans:
                TESTP((nntile::TransOp(value)));
                break;
            default:
                TESTN((nntile::TransOp(value)));
        }
    }
    return 0;
}

