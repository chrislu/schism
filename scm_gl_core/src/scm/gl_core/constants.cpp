
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "constants.h"

#include <cassert>

namespace scm {
namespace gl {

// debugging //////////////////////////////////////////////////////////////////////////////////////

const char*
debug_source_string(debug_source s)
{
    assert(DEBUG_SOURCE_API <= s && s < DEBUG_SOURCE_COUNT);

    switch (s) {
    case DEBUG_SOURCE_API:              return ("api");break;
    case DEBUG_SOURCE_WINDOW_SYSTEM:    return ("window system");break;
    case DEBUG_SOURCE_SHADER_COMPILER:  return ("shader compiler");break;
    case DEBUG_SOURCE_THIRD_PARTY:      return ("third party");break;
    case DEBUG_SOURCE_APPLICATION:      return ("application");break;
    case DEBUG_SOURCE_OTHER:            return ("other");break;
    default: break;
    }

    return ("unknown debug_source");
}

const char*
debug_type_string(debug_type s)
{
    assert(DEBUG_TYPE_ERROR <= s && s < DEBUG_TYPE_COUNT);

    switch (s) {
    case DEBUG_TYPE_ERROR:                  return ("error");break;
    case DEBUG_TYPE_DEPRECATED_BEHAVIOR:    return ("deprecated behavior");break;
    case DEBUG_TYPE_UNDEFINED_BEHAVIOR:     return ("undefined behavior");break;
    case DEBUG_TYPE_PORTABILITY:            return ("portability");break;
    case DEBUG_TYPE_PERFORMANCE:            return ("performance");break;
    case DEBUG_TYPE_OTHER:                  return ("other");break;
    default: break;
    }

    return ("unknown debug_type");
}

const char*
debug_severity_string(debug_severity s)
{
    assert(DEBUG_SEVERITY_HIGH <= s && s < DEBUG_SEVERITY_COUNT);

    switch (s) {
    case DEBUG_SEVERITY_HIGH:   return ("high");break;
    case DEBUG_SEVERITY_MEDIUM: return ("medium");break;
    case DEBUG_SEVERITY_LOW:    return ("low");break;
    default: break;
    }

    return ("unknown debug_severity");
}

// shader /////////////////////////////////////////////////////////////////////////////////////////

const char*
shader_stage_string(shader_stage s)
{
    assert(STAGE_VERTEX_SHADER <= s && s < SHADER_STAGE_COUNT);

    switch (s) {
    case STAGE_VERTEX_SHADER:          return ("vertex shader");break;
    case STAGE_GEOMETRY_SHADER:        return ("geometry shader");break;
    case STAGE_FRAGMENT_SHADER:        return ("fragment shader");break;

#if SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400
    case STAGE_TESS_EVALUATION_SHADER: return ("tesselation evaluation shader");break;
    case STAGE_TESS_CONTROL_SHADER:    return ("tesselation control shader");break;
#endif // SCM_GL_CORE_OPENGL_CORE_VERSION >= SCM_GL_CORE_OPENGL_CORE_VERSION_400

    default: break;
    }

    return ("unknown shader_stage");
}

} // namespace gl
} // namespace scm
