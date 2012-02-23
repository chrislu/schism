
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_TEXTURE_BUFFER_H_INCLUDED
#define SCM_GL_CORE_TEXTURE_BUFFER_H_INCLUDED

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/buffer_objects/buffer_objects_fwd.h>
#include <scm/gl_core/texture_objects/texture.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

struct __scm_export(gl_core) texture_buffer_desc
{
    texture_buffer_desc(const data_format   in_format,
                        const buffer_ptr&   in_buffer);

    bool operator==(const texture_buffer_desc& rhs) const;
    bool operator!=(const texture_buffer_desc& rhs) const;

    data_format     _format;
    buffer_ptr      _buffer;
}; // struct texture_buffer_desc

class __scm_export(gl_core) texture_buffer : public texture
{
public:
    virtual ~texture_buffer();

    const texture_buffer_desc&  descriptor() const;
    void                        print(std::ostream& os) const {};

protected:
    texture_buffer(render_device&             in_device,
                   const texture_buffer_desc& in_desc);

protected:
    texture_buffer_desc         _descriptor;

private:

    friend class render_device;
    friend class render_context;
}; // class texture_buffer

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_TEXTURE_BUFFER_H_INCLUDED
