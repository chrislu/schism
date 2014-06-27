
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_OCCLUSION_QUERY_H_INCLUDED
#define SCM_GL_CORE_OCCLUSION_QUERY_H_INCLUDED

#include <scm/core/numeric_types.h>

#include <scm/gl_core/constants.h>
#include <scm/gl_core/query_objects/query.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) occlusion_query : public query
{
public:
    virtual ~occlusion_query();

    scm::uint64     result() const;

protected:
    occlusion_query(render_device& in_device, const occlusion_query_mode in_oq_mode);

    void            collect(const render_context& in_context);

protected:
    scm::uint64     _result;

private:
    friend class render_device;
    friend class render_context;
}; // class query

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_OCCLUSION_QUERY_H_INCLUDED
