
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_CONTEXT_BINDABLE_OBJECT_H_INCLUDED
#define SCM_GL_CORE_CONTEXT_BINDABLE_OBJECT_H_INCLUDED

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) context_bindable_object
{
public:
    context_bindable_object();
    virtual ~context_bindable_object();

public:
    unsigned                object_id() const       { return (_gl_object_id); };
    unsigned                object_target() const   { return (_gl_object_target); };
    unsigned                object_binding() const  { return (_gl_object_binding); };

protected:
    unsigned                _gl_object_id;
    unsigned                _gl_object_target;
    unsigned                _gl_object_binding;

}; // class context_bindable_object

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_CONTEXT_BINDABLE_OBJECT_H_INCLUDED
