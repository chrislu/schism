
#ifndef SCM_GL_CORE_STREAM_CAPTURE_H_INCLUDED
#define SCM_GL_CORE_STREAM_CAPTURE_H_INCLUDED

#include <list>
#include <string>
#include <vector>

#include <boost/variant.hpp>

#include <scm/gl_core/constants.h>
#include <scm/gl_core/data_types.h>
#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/shader_objects/shader_objects_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) stream_capture
{
public:
    typedef enum {
        skip_1_float    = 0x00,
        skip_2_float,
        skip_3_float,
        skip_4_float
    } skip_components_type;

    typedef boost::variant<std::string, skip_components_type>   capture_element;
    typedef std::list<capture_element>                          capture_varyings_list;
    typedef std::vector<capture_varyings_list>                  stream_captures_array;

public:
    stream_capture();
    stream_capture(const output_stream stream, const std::string& varying_name);
    stream_capture(const output_stream stream, const skip_components_type skip_components);
    /*virtual*/ ~stream_capture();

    stream_capture&                 operator()(const output_stream stream, const std::string& varying_name);
    stream_capture&                 operator()(const output_stream stream, const skip_components_type skip_components);

    void                            append_capture(const output_stream stream, const std::string& varying_name);
    void                            append_capture(const output_stream stream, const skip_components_type skip_components);

    bool                            empty() const;
    unsigned                        max_used_stream() const;
    bool                            interleaved_streams() const;
    int                             captures_count() const;

    const capture_varyings_list&    captures(const output_stream stream) const;

protected:
    stream_captures_array           _stream_captures;
    unsigned                        _max_used_stream;
    int                             _captures_count;

}; // class stream_capture

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_STREAM_CAPTURE_H_INCLUDED
