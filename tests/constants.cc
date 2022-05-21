#include "nntile/constants.hh"
#include "testing.hh"

using namespace nntile;

int main(int argc, char **argv)
{
    TESTP((TransOp(TransOp::NoTrans)));
    TESTP((TransOp(TransOp::Trans)));
    for(int i = -10; i < 10; ++i)
    {
        auto value = static_cast<enum TransOp::Value>(i);
        switch(value)
        {
            case TransOp::NoTrans:
            case TransOp::Trans:
                TESTP((TransOp(value)));
                break;
            default:
                TESTN((TransOp(value)));
        }
    }
    return 0;
}

