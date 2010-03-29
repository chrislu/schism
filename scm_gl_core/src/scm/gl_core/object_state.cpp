
#include "object_state.h"

namespace scm {
namespace gl {

std::string
object_state::state_string() const
{
    std::string out_string;

    switch (_state) {
        case OS_OK:                         out_string.assign("OS_OK");                         break;
        case OS_BAD:                        out_string.assign("OS_BAD");                        break;
        case OS_FAIL:                       out_string.assign("OS_FAIL");                       break;
        case OS_ERROR_INVALID_ENUM:         out_string.assign("OS_ERROR_INVALID_ENUM");         break;
        case OS_ERROR_INVALID_VALUE:        out_string.assign("OS_ERROR_INVALID_VALUE");        break;
        case OS_ERROR_INVALID_OPERATION:    out_string.assign("OS_ERROR_INVALID_OPERATION");    break;
        case OS_ERROR_OUT_OF_MEMORY:        out_string.assign("OS_ERROR_OUT_OF_MEMORY");        break;
        case OS_ERROR_SHADER_COMPILE:       out_string.assign("OS_ERROR_SHADER_COMPILE");       break;
        case OS_ERROR_SHADER_LINK:          out_string.assign("OS_ERROR_SHADER_LINK");          break;
        case OS_ERROR_UNKNOWN:              out_string.assign("OS_ERROR_UNKNOWN");              break;
        default:                            out_string.assign("unknown state");                 break;
    };
    return (out_string);
}

} // namespace gl
} // namespace scm
