
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_STREAM_CAPTURE_H_INCLUDED
#define SCM_GL_CORE_STREAM_CAPTURE_H_INCLUDED

#include <list>
#include <string>
#include <vector>

#include <boost/variant.hpp>

#include <scm/core/memory.h>

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
    typedef std::list<capture_element>                          captures_list;

public:
    stream_capture();
    virtual ~stream_capture();

    virtual bool                is_interleaved() const = 0;

    bool                        empty() const;
    int                         size() const;

    const captures_list&        captures() const;

protected:
    captures_list               _elements;

}; // class stream_capture

class __scm_export(gl_core) separate_stream_capture : public stream_capture
{
public:
    separate_stream_capture(const std::string& varying_name);
    virtual ~separate_stream_capture();

    bool                        is_interleaved() const;

}; // class separate_stream_capture

class __scm_export(gl_core) interleaved_stream_capture : public stream_capture
{
public:
    interleaved_stream_capture(const std::string& varying_name);
    interleaved_stream_capture(const skip_components_type& skip_components);
    virtual ~interleaved_stream_capture();

    interleaved_stream_capture& operator()(const std::string& varying_name);
    interleaved_stream_capture& operator()(const skip_components_type& skip_components);

    bool                        is_interleaved() const;
    bool                        has_skipped_components() const;

private:
    bool                        _has_skipped_components;

}; // class interleaved_stream_capture

class __scm_export(gl_core) stream_capture_array
{
protected:
    typedef shared_ptr<stream_capture>  capture_ptr;
    typedef std::vector<capture_ptr>    stream_capture_vector;

public:
    stream_capture_array();
    stream_capture_array(const std::string&                varying_name);
    stream_capture_array(const separate_stream_capture&    capture);
    stream_capture_array(const interleaved_stream_capture& capture);
    /*virtual*/ ~stream_capture_array();

    stream_capture_array&       operator()(const std::string&                varying_name);
    stream_capture_array&       operator()(const separate_stream_capture&    capture);
    stream_capture_array&       operator()(const interleaved_stream_capture& capture);

    void                        append_capture(const std::string&                varying_name); // appends a separate capture object
    void                        append_capture(const separate_stream_capture&    capture);
    void                        append_capture(const interleaved_stream_capture& capture);

    bool                        empty() const;
    int                         used_streams() const;
    bool                        interleaved_streams() const;
    bool                        interleaved_skipped_components() const;
    int                         captures_count() const;

    const stream_capture&       stream_captures(const int stream) const;

protected:
    stream_capture_vector       _stream_captures;
    int                         _captures_count;
    bool                        _interleaved_streams;
    bool                        _interleaved_skipped_components;

}; // class stream_capture_array

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_STREAM_CAPTURE_H_INCLUDED
