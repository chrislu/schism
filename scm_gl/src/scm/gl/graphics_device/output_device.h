
#ifndef SCM_GL_OUTPUT_DEVICE_OUTPUT_DEVICE_H_INCLUDED
#define SCM_GL_OUTPUT_DEVICE_OUTPUT_DEVICE_H_INCLUDED

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) output_device_descriptor
{
public:
    unsigned            _device_id;
    unsigned            _screen_id;

    output_device_descriptor(unsigned dev, unsigned scr);

    // unix style 0:0 is first device, first monitor etc.

}; // class output_device_descriptor

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_OUTPUT_DEVICE_OUTPUT_DEVICE_H_INCLUDED
