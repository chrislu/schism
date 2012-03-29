
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "context_format.h"

#include <algorithm>

namespace scm {
namespace gl_classic {

context_format::context_format()
   : _color_bits(0),
    _depth_bits(0),
    _alpha_bits(0),
    _stencil_bits(0),
    _max_aux_buffers(0),
    _max_samples(0),
    _double_buffer(false),
    _stereo(false),
    _debug(false),
    _forward_compatible(false),
    _compatibility_profile(true),
    _version_major(1),
    _version_minor(0)
{
}

context_format::context_format(const context_format& fmt)
  : _color_bits(fmt._color_bits),
    _depth_bits(fmt._depth_bits),
    _alpha_bits(fmt._alpha_bits),
    _stencil_bits(fmt._stencil_bits),
    _max_aux_buffers(fmt._max_aux_buffers),
    _max_samples(fmt._max_samples),
    _double_buffer(fmt._double_buffer),
    _stereo(fmt._stereo),
    _debug(fmt._debug),
    _forward_compatible(fmt._forward_compatible),
    _compatibility_profile(fmt._compatibility_profile),
    _version_major(fmt._version_major),
    _version_minor(fmt._version_minor)
{
}

context_format::~context_format()
{
}

context_format&
context_format::operator=(const context_format& rhs)
{
    context_format tmp(rhs);

    swap(tmp);

    return (*this);
}

void
context_format::swap(context_format& fmt)
{
    std::swap(_color_bits,              fmt._color_bits);
    std::swap(_depth_bits,              fmt._depth_bits);
    std::swap(_alpha_bits,              fmt._alpha_bits);
    std::swap(_stencil_bits,            fmt._stencil_bits);
    std::swap(_max_aux_buffers,         fmt._max_aux_buffers);
    std::swap(_max_samples,             fmt._max_samples);
    std::swap(_double_buffer,           fmt._double_buffer);
    std::swap(_stereo,                  fmt._stereo);
    std::swap(_debug,                   fmt._debug);
    std::swap(_forward_compatible,      fmt._forward_compatible);
    std::swap(_compatibility_profile,   fmt._compatibility_profile);
    std::swap(_version_major,           fmt._version_major);
    std::swap(_version_minor,           fmt._version_minor);
}

bool context_format::operator==(const context_format& fmt) const
{
    bool tmp_ret = true;

    tmp_ret = tmp_ret && (_color_bits               == fmt._color_bits);
    tmp_ret = tmp_ret && (_depth_bits               == fmt._depth_bits);
    tmp_ret = tmp_ret && (_alpha_bits               == fmt._alpha_bits);
    tmp_ret = tmp_ret && (_stencil_bits             == fmt._stencil_bits);
    tmp_ret = tmp_ret && (_max_aux_buffers          == fmt._max_aux_buffers);
    tmp_ret = tmp_ret && (_max_samples              == fmt._max_samples);
    tmp_ret = tmp_ret && (_double_buffer            == fmt._double_buffer);
    tmp_ret = tmp_ret && (_stereo                   == fmt._stereo);
    tmp_ret = tmp_ret && (_debug                    == fmt._debug);
    tmp_ret = tmp_ret && (_forward_compatible       == fmt._forward_compatible);
    tmp_ret = tmp_ret && (_compatibility_profile    == fmt._compatibility_profile);
    tmp_ret = tmp_ret && (_version_major            == fmt._version_major);
    tmp_ret = tmp_ret && (_version_minor            == fmt._version_minor);

    return (tmp_ret);
}

bool context_format::operator!=(const context_format& fmt) const
{
    return (!(*this == fmt));
}

/*static*/ const context_format&
context_format::null_format()
{
    static context_format nullfmt;

    return (nullfmt);
}

/*static*/ const context_format&
context_format::default_format()
{
    static context_format deffmt;

    deffmt._color_bits      = 32;
    deffmt._depth_bits      = 24;
    deffmt._alpha_bits      = 8;
    deffmt._stencil_bits    = 8;
    deffmt._double_buffer   = true;

    return (deffmt);
}

} // namespace gl_classic
} // namespace scm
