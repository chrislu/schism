
#include "system_exception.h"

using namespace scm::core;

namespace {
    static const std::string      system_error_intro = std::string("critical system error: ");
}

system_exception::system_exception(const std::string& msg)
: std::runtime_error(system_error_intro + msg)
{
}
