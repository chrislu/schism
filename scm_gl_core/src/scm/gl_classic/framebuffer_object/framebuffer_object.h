
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef FRAMEBUFFER_OBJECT_H_INCLUDED
#define FRAMEBUFFER_OBJECT_H_INCLUDED

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/luabind_warning_disable.h>

namespace scm {
namespace gl_classic {

class renderbuffer;

class __scm_export(gl_core) framebuffer_object
{
public:
    framebuffer_object();
    virtual ~framebuffer_object();

    bool    bind();

    bool    bind_to_draw();
    bool    bind_to_read();

    static void    unbind();

    static void    unbind_from_read();
    static void    unbind_from_draw();

protected:

private:
    framebuffer_object(const framebuffer_object&);
    const framebuffer_object& operator=(const framebuffer_object&);

}; // class framebuffer_object

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/luabind_warning_enable.h>

#endif // FRAMEBUFFER_OBJECT_H_INCLUDED
