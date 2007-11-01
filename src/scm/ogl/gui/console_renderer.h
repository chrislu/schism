
#ifndef GL_CONSOLE_RENDERER_H_INCLUDED
#define GL_CONSOLE_RENDERER_H_INCLUDED

#include <list>
#include <string>

#include <scm/ogl/gui/text_box.h>
#include <scm/core/console/console_output_listener.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {
namespace gui {

class __scm_export(ogl) console_renderer : public con::console_output_listener,
                                           public text_box
{
protected:
public:
    console_renderer();
    virtual ~console_renderer();

protected:
    void                            update(const std::string&               /*update_buffer*/,
                                           const con::console_out_stream&   /*stream_source*/);

}; // class console_renderer

} // namespace gui
} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // GL_CONSOLE_RENDERER_H_INCLUDED
