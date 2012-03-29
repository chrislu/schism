
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "stream_capture.h"

#include <cassert>
#include <limits>

#include <scm/core/math.h>

namespace {
} // namespace

namespace scm {
namespace gl {

// stream_capture /////////////////////////////////////////////////////////////////////////////////
stream_capture::stream_capture()
{
}

stream_capture::~stream_capture()
{
}

bool
stream_capture::empty() const
{
    return _elements.empty();
}

int
stream_capture::size() const
{
    assert(_elements.size() < (std::numeric_limits<int>::max)());
    return static_cast<int>(_elements.size());
}

const stream_capture::captures_list&
stream_capture::captures() const
{
    return _elements;
}

// separate_stream_capture ////////////////////////////////////////////////////////////////////////
separate_stream_capture::separate_stream_capture(const std::string& varying_name)
{
    _elements.push_back(varying_name);
}

separate_stream_capture::~separate_stream_capture()
{
}

bool
separate_stream_capture::is_interleaved() const
{
    return false;
}

// interleaved_stream_capture /////////////////////////////////////////////////////////////////////
interleaved_stream_capture::interleaved_stream_capture(const std::string& varying_name)
  : _has_skipped_components(false)
{
    _elements.push_back(varying_name);
}

interleaved_stream_capture::interleaved_stream_capture(const skip_components_type& skip_components)
  : _has_skipped_components(true)
{
    _elements.push_back(skip_components);
}

interleaved_stream_capture::~interleaved_stream_capture()
{
}

bool
interleaved_stream_capture::is_interleaved() const
{
    return true;
}

bool
interleaved_stream_capture::has_skipped_components() const
{
    return _has_skipped_components;
}

interleaved_stream_capture&
interleaved_stream_capture::operator()(const std::string& varying_name)
{
    _elements.push_back(varying_name);
    return *this;
}

interleaved_stream_capture&
interleaved_stream_capture::operator()(const skip_components_type& skip_components)
{
    _elements.push_back(skip_components);
    _has_skipped_components = true;
    return *this;
}

// stream_capture_array ///////////////////////////////////////////////////////////////////////////
stream_capture_array::stream_capture_array()
  : _captures_count(0)
  , _interleaved_streams(false)
  , _interleaved_skipped_components(false)
{
}

stream_capture_array::stream_capture_array(const std::string& varying_name)
  : _captures_count(0)
  , _interleaved_streams(false)
  , _interleaved_skipped_components(false)
{
    append_capture(varying_name);
}

stream_capture_array::stream_capture_array(const separate_stream_capture& capture)
  : _captures_count(0)
  , _interleaved_streams(false)
  , _interleaved_skipped_components(false)
{
    append_capture(capture);
}

stream_capture_array::stream_capture_array(const interleaved_stream_capture& capture)
  : _captures_count(0)
  , _interleaved_streams(true)
  , _interleaved_skipped_components(capture.has_skipped_components())
{
    append_capture(capture);
}

stream_capture_array::~stream_capture_array()
{
}

stream_capture_array&
stream_capture_array::operator()(const std::string& varying_name)
{
    append_capture(varying_name);
    return *this;
}

stream_capture_array&
stream_capture_array::operator()(const separate_stream_capture& capture)
{
    append_capture(capture);
    return *this;
}

stream_capture_array&
stream_capture_array::operator()(const interleaved_stream_capture& capture)
{
    append_capture(capture);
    return *this;
}

void
stream_capture_array::append_capture(const std::string& varying_name) // appends a separate capture object
{
    _stream_captures.push_back(scm::make_shared<separate_stream_capture>(varying_name));
    _captures_count += 1;
}

void
stream_capture_array::append_capture(const separate_stream_capture& capture)
{
    _stream_captures.push_back(make_shared<separate_stream_capture>(capture));
    assert(capture.size() == 1);
    _captures_count += capture.size();
}

void
stream_capture_array::append_capture(const interleaved_stream_capture& capture)
{
    _stream_captures.push_back(make_shared<interleaved_stream_capture>(capture));

    _interleaved_streams            = true;
    _interleaved_skipped_components = capture.has_skipped_components();
    _captures_count                += capture.size();
}

bool
stream_capture_array::empty() const
{
    bool captures_empty = true;

    for (int i = 0; i < _stream_captures.size(); ++i) {
        captures_empty = captures_empty && _stream_captures[i]->empty();
    }

    return captures_empty;
}

int
stream_capture_array::used_streams() const
{
    assert(_stream_captures.size() < (std::numeric_limits<int>::max)());
    return static_cast<int>(_stream_captures.size());
}

bool
stream_capture_array::interleaved_streams() const
{
    return _interleaved_streams;
}

bool
stream_capture_array::interleaved_skipped_components() const
{
    return _interleaved_skipped_components;
}

int
stream_capture_array::captures_count() const
{
    return _captures_count;
}

const stream_capture&
stream_capture_array::stream_captures(const int stream) const
{
    assert(0 <= stream && stream < _stream_captures.size());
    assert(_stream_captures[stream]);
    return *_stream_captures[stream];
}

} // namespace gl
} // namespace scm
