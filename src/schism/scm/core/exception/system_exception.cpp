
#include "system_exception.h"

namespace {
    static const std::string      system_error_intro = std::string("critical system error: ");
}

namespace scm {
namespace core {

system_exception::system_exception(const std::string& msg)
: std::runtime_error(system_error_intro + msg)
{
}

} // namespace core
} // namespace scm
