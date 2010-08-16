
#include "constants.h"

#include <cassert>

namespace scm {
namespace gl {

const char*
shader_stage_string(shader_stage s)
{
    assert(STAGE_VERTEX_SHADER <= s && s < SHADER_STAGE_COUNT);

    switch (s) {
    case STAGE_VERTEX_SHADER:          return ("VERTEX_SHADER");break;
    case STAGE_GEOMETRY_SHADER:        return ("GEOMETRY_SHADER");break;
    case STAGE_FRAGMENT_SHADER:        return ("FRAGMENT_SHADER");break;

#if SCM_GL_CORE_OPENGL_40
    case STAGE_TESS_EVALUATION_SHADER: return ("TESS_EVALUATION_SHADER");break;
    case STAGE_TESS_CONTROL_SHADER:    return ("TESS_CONTROL_SHADER");break;
#endif

    default: break;
    }

    return ("unknown shader stage");
}

} // namespace gl
} // namespace scm
