
#ifndef GL_CONSOLE_RENDERER_H_INCLUDED
#define GL_CONSOLE_RENDERER_H_INCLUDED

#include <list>
#include <string>

#include <scm/core/console/console_output_listener.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(ogl) console_renderer : public con::console_output_listener
{
protected:
    typedef std::list<std::string>          string_list;

public:
    console_renderer();
    virtual ~console_renderer();

protected:
    void                            update(const std::string&               /*update_buffer*/,
                                           const con::console_out_stream&   /*stream_source*/);

    string_list                     _lines;

}; // class console_renderer

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // GL_CONSOLE_RENDERER_H_INCLUDED
