
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_SCOPED_BUFFER_MAP_H_INCLUDED
#define SCM_GL_CORE_SCOPED_BUFFER_MAP_H_INCLUDED

#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>

#include <scm/gl_core/constants.h>
#include <scm/gl_core/gl_core_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) scoped_buffer_map
{
public:
    explicit scoped_buffer_map(const render_context_ptr& in_context,
                               const buffer_ptr&         in_buffer,
                               const access_mode         in_access);
    explicit scoped_buffer_map(const render_context_ptr& in_context,
                               const buffer_ptr&         in_buffer,
                                     scm::size_t         in_offset,
                                     scm::size_t         in_size,
                               const access_mode         in_access);
    ~scoped_buffer_map();

                                operator bool() const;
    bool                        valid() const;
    uint8*const                 data_ptr() const;

private:
    const render_context_ptr    _context;
    const buffer_ptr            _buffer;

    uint8*                      _data;

private: // declared, never defined
    scoped_buffer_map(const scoped_buffer_map&);
    scoped_buffer_map& operator=(const scoped_buffer_map&);

}; // class scoped_buffer_map

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_BUFFER_H_INCLUDED
