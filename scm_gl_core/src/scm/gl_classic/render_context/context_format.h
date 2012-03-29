
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CONTEXT_FORMAT_H_INCLUDED
#define SCM_GL_CONTEXT_FORMAT_H_INCLUDED

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) context_format
{
public:
    context_format();
    context_format(const context_format& fmt);
    /*virtual*/ ~context_format();

    context_format&         operator=(const context_format& rhs);
    void                    swap(context_format& fmt);

    bool                    operator==(const context_format& fmt) const;
    bool                    operator!=(const context_format& fmt) const;

    unsigned                color_bits() const                 { return (_color_bits); }
    unsigned                depth_bits() const                 { return (_depth_bits); }
    unsigned                alpha_bits() const                 { return (_alpha_bits); }
    unsigned                stencil_bits() const               { return (_stencil_bits); }
    unsigned                max_aux_buffers() const            { return (_max_aux_buffers); }
    unsigned                max_samples() const                { return (_max_samples); }
    bool                    double_buffer() const              { return (_double_buffer); }
    bool                    stereo() const                     { return (_stereo); }

    bool                    debug() const                      { return (_debug); }
    bool                    forward_compatible() const         { return (_forward_compatible); }
    bool                    compatibility_profile() const      { return (_compatibility_profile); }
    int                     version_major() const              { return (_version_major); }
    int                     version_minor() const              { return (_version_minor); }

    void                    color_bits(unsigned int v)         { _color_bits = v; }
    void                    depth_bits(unsigned int v)         { _depth_bits = v; }
    void                    alpha_bits(unsigned int v)         { _alpha_bits = v; }
    void                    stencil_bits(unsigned int v)       { _stencil_bits = v; }
    void                    num_aux_buffers(unsigned int v)    { _max_aux_buffers = v; }
    void                    num_samples(unsigned int v)        { _max_samples = v; }
    void                    double_buffer(bool v)              { _double_buffer = v; }
    void                    stereo(bool v)                     { _stereo = v; }

    void                    debug(bool v)                      { _debug = v; }
    void                    forward_compatible(bool v)         { _forward_compatible = v; }
    void                    compatibility_profile(bool v)      { _compatibility_profile = v; }
    void                    version_major(int v)               { _version_major = v; }
    void                    version_minor(int v)               { _version_minor = v; }

    static const context_format& null_format();
    static const context_format& default_format();

private:

    unsigned                _color_bits;
    unsigned                _depth_bits;
    unsigned                _alpha_bits;
    unsigned                _stencil_bits;
    unsigned                _max_aux_buffers;
    unsigned                _max_samples;
    bool                    _double_buffer;
    bool                    _stereo;

    bool                    _debug;
    bool                    _forward_compatible;
    bool                    _compatibility_profile;
    int                     _version_major;
    int                     _version_minor;

}; // class context_format

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CONTEXT_FORMAT_H_INCLUDED
