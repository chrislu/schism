
#ifndef SCM_GL_CORE_OBJECT_STATE_H_INCLUDED
#define SCM_GL_CORE_OBJECT_STATE_H_INCLUDED

#include <string>

#include <scm/core/platform/platform.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) object_state
{
public:
    static const unsigned OS_OK                         = 0x0000;
    static const unsigned OS_BAD                        = 0x0001;
    static const unsigned OS_FAIL                       = 0x0002;

    static const unsigned OS_ERROR_INVALID_ENUM         = 0x0010 | OS_FAIL;
    static const unsigned OS_ERROR_INVALID_VALUE        = 0x0020 | OS_FAIL;
    static const unsigned OS_ERROR_INVALID_OPERATION    = 0x0040 | OS_FAIL;
    static const unsigned OS_ERROR_OUT_OF_MEMORY        = 0x0080 | OS_FAIL;
    static const unsigned OS_ERROR_SHADER_COMPILE       = 0x0100 | OS_FAIL;
    static const unsigned OS_ERROR_SHADER_LINK          = 0x0200 | OS_FAIL;

    static const unsigned OS_ERROR_UNKNOWN              = 0x1000 | OS_FAIL;

    static const unsigned OS_STATE_MASK                 = 0x1fff;

public:
    object_state() : _state(OS_OK) {}

                operator bool() const   { return (ok()); }
    bool        operator!() const       { return (fail()); }
    bool        ok() const              { return (OS_OK == _state); }
    bool        bad() const             { return (0 != (_state & OS_BAD)); }
    bool        fail() const            { return (0 != (_state & (OS_BAD | OS_FAIL))); }

    unsigned    get() const             { return (_state); }
    void        set(unsigned s)         { _state = (s & OS_STATE_MASK); }
    void        clear()                 { set(OS_OK); }

    std::string state_string() const;

private:
    unsigned    _state;
};

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_OBJECT_STATE_H_INCLUDED
