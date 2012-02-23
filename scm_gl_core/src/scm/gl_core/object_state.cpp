
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

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

        case OS_ERROR_FRAMEBUFFER_UNDEFINED:                     out_string.assign("OS_ERROR_FRAMEBUFFER_UNDEFINED");                     break;
        case OS_ERROR_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:         out_string.assign("OS_ERROR_FRAMEBUFFER_INCOMPLETE_ATTACHMENT");         break;
        case OS_ERROR_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: out_string.assign("OS_ERROR_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT"); break;
        case OS_ERROR_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:        out_string.assign("OS_ERROR_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER");        break;
        case OS_ERROR_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:        out_string.assign("OS_ERROR_FRAMEBUFFER_INCOMPLETE_READ_BUFFER");        break;
        case OS_ERROR_FRAMEBUFFER_UNSUPPORTED:                   out_string.assign("OS_ERROR_FRAMEBUFFER_UNSUPPORTED");                   break;
        case OS_ERROR_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:        out_string.assign("OS_ERROR_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE");        break;
        case OS_ERROR_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:      out_string.assign("OS_ERROR_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS");      break;
        
        default:                            out_string.assign("unknown state");                 break;
    };
    return (out_string);
}

} // namespace gl
} // namespace scm
