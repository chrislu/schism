
#include "stream_capture.h"

#include <cassert>

#include <scm/core/math.h>

namespace {
} // namespace

namespace scm {
namespace gl {

stream_capture::stream_capture()
  : _stream_captures(static_cast<size_t>(OUTPUT_STREAM_COUNT))
  , _max_used_stream(0)
  , _captures_count(0)
{
    assert(_stream_captures.size() == OUTPUT_STREAM_COUNT);
}

stream_capture::stream_capture(const output_stream stream, const std::string& varying_name)
  : _stream_captures(static_cast<size_t>(OUTPUT_STREAM_COUNT))
  , _max_used_stream(0)
  , _captures_count(0)
{
    assert(_stream_captures.size() == OUTPUT_STREAM_COUNT);

    append_capture(stream, varying_name);
}

stream_capture::stream_capture(const output_stream stream, const skip_components_type skip_components)
  : _stream_captures(static_cast<size_t>(OUTPUT_STREAM_COUNT))
  , _max_used_stream(0)
  , _captures_count(0)
{
    assert(_stream_captures.size() == OUTPUT_STREAM_COUNT);

    append_capture(stream, skip_components);
}

stream_capture::~stream_capture()
{
}

stream_capture&
stream_capture::operator()(const output_stream stream, const std::string& varying_name)
{
    append_capture(stream, varying_name);
    return *this;
}

stream_capture&
stream_capture::operator()(const output_stream stream, const skip_components_type skip_components)
{
    append_capture(stream, skip_components);
    return *this;
}

void
stream_capture::append_capture(const output_stream stream, const std::string& varying_name)
{
    assert(static_cast<int>(stream) < _stream_captures.size());

    _stream_captures[stream].push_back(varying_name);
    _max_used_stream  = math::max<unsigned>(_max_used_stream, stream);
    _captures_count  += 1;
}

void
stream_capture::append_capture(const output_stream stream, const skip_components_type skip_components)
{
    assert(static_cast<int>(stream) < _stream_captures.size());

    _stream_captures[stream].push_back(skip_components);
    _max_used_stream = math::max<unsigned>(_max_used_stream, stream);
    _captures_count  += 1;
}

bool
stream_capture::empty() const
{
    bool captures_empty = true;

    for (int i = 0; i < _stream_captures.size(); ++i) {
        captures_empty = captures_empty && _stream_captures[i].empty();
    }

    return captures_empty;
}

unsigned
stream_capture::max_used_stream() const
{
    return _max_used_stream;
}

bool
stream_capture::interleaved_streams() const
{
    bool captures_interleaved = false;

    for (int i = 0; i < _stream_captures.size(); ++i) {
        captures_interleaved = captures_interleaved || (_stream_captures[i].size() > 1);
    }

    return captures_interleaved;
}

int
stream_capture::captures_count() const
{
    return _captures_count;
}

const stream_capture::capture_varyings_list&
stream_capture::captures(const output_stream stream) const
{
    assert(static_cast<int>(stream) < _stream_captures.size());

    return _stream_captures[stream];
}

} // namespace gl
} // namespace scm
